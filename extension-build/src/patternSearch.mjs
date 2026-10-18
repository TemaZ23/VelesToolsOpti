// Step-by-step (pattern) search for the strategy optimizer.
//
// The genetic search changed many parameters at random at once and, with ~100 evaluations
// in 20–30 dimensions, mostly produced variants that trade less and are worse than a strategy
// that has already been tuned by hand. This module replaces it with a deterministic local
// search that is sample-efficient for expensive backtests:
//
//   1. scan     — every parameter (and a few "macro" directions such as "all grid indents
//                 wider") is moved one step up and one step down from the current point;
//                 this also gives the user a sensitivity map;
//   2. climb    — improving moves are combined (all at once, or greedily one by one),
//                 then each accepted direction is followed while it keeps improving;
//   3. pattern  — the whole move of the round is repeated once more (Hooke–Jeeves);
//   4. refine   — the step is halved and only parameters that matter are scanned again.
//
// Scoring helpers implement the portfolio objective:
//   • "better on both after deposit rescale": profit and drawdown of a portfolio scale linearly
//     with the bot deposit, so a variant is compared with the original after choosing the bot
//     deposit (within ±maxChange) that balances the profit gain and the drawdown gain;
//   • fold consistency: the improvement must hold in most time periods and coin groups of the
//     search set, not only on aggregate.
//
// Everything here is pure and has no dependency on the app bundle (helpers are injected).

const EPS = 1e-9;
const DAY_MS = 24 * 60 * 60 * 1000;

export const PS_DEFAULTS = Object.freeze({
  /** First scan step, as a share of each parameter's range. */
  initialStep: 0.125,
  /** Smallest step before the search stops refining. */
  minStep: 1 / 32,
  /** Minimal score gain for a move to be accepted. */
  margin: 0.0025,
  /** How many extra steps to take along an accepted direction. */
  maxLineSteps: 3,
  /** Alternative indicators tested per filter in the first scan. */
  choiceSamples: 2,
  /** Stop after this many rounds in a row without improvement (each one halves the step). */
  stallRounds: 2,
  /** Moves whose effect is below this are not rescanned. */
  insensitive: 0.002,
  /** Moves that made things worse by more than this are not rescanned. */
  harmful: 0.03,
  /** Share of the budget kept for combining moves after the first scan. */
  reserveShare: 0.15,
});

/** @param {number} value @param {number} min @param {number} max */
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

/** Locale-independent string order (keeps the search deterministic everywhere). */
const compareText = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

/** @param {number} value @param {number} digits */
const roundTo = (value, digits) => {
  const factor = 10 ** digits;
  const rounded = Math.round(value * factor) / factor;
  return rounded === 0 ? 0 : rounded;
};

/** @param {number[]} values */
export const psMedian = (values) => {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
};

/** FNV-1a hash of a string — a stable seed for the search RNG. @param {string} text */
export const psHashSeed = (text) => {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

/** Deterministic RNG (mulberry32) so that an interrupted search replays identically. @param {number} seed */
export const psCreateRng = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** @param {Record<string, number>} genome @param {{id: string, baseValue: number}} dim */
export const psValue = (genome, dim) => {
  const value = genome[dim.id];
  return typeof value === 'number' && Number.isFinite(value) ? value : dim.baseValue;
};

/** Filter swap / added filter: the value is an index into `choices`. */
export const psIsChoice = (dim) =>
  (dim.kind === 'conditionSwap' || dim.kind === 'conditionAdd') && Array.isArray(dim.choices);

const dimensionMaps = new WeakMap();
const dimensionMap = (space) => {
  let map = dimensionMaps.get(space);
  if (!map) {
    map = new Map(space.dimensions.map((dim) => [dim.id, dim]));
    dimensionMaps.set(space, map);
  }
  return map;
};

/** True when both genomes have the same value for every search dimension. */
export const psSameGenome = (space, a, b) =>
  space.dimensions.every((dim) => Math.abs(psValue(a, dim) - psValue(b, dim)) < 1e-9);

/** Scope of a filter threshold: entry, order (grid), takeProfit or stopLoss. */
const conditionScope = (dim) => {
  const match = /^cond:v:([a-zA-Z]+)/.exec(dim.id);
  if (!match) {
    return 'entry';
  }
  if (match[1].startsWith('order')) {
    return 'order';
  }
  return match[1];
};

const stepPriority = (dim) => {
  switch (dim.kind) {
    case 'takeProfit':
      return 10;
    case 'stopLoss':
      return 12;
    case 'stopLossConditional':
      return 14;
    case 'pullUp':
      return 16;
    case 'conditionValue': {
      const scope = conditionScope(dim);
      return scope === 'entry' ? 30 : scope === 'order' ? 34 : 38;
    }
    case 'orderIndent':
      return 50 + (dim.orderIndex ?? 0) / 100;
    case 'conditionInterval':
      return 60;
    case 'orderVolume':
      return 80 + (dim.orderIndex ?? 0) / 100;
    case 'leverage':
      return 90;
    default:
      return 70;
  }
};

const shortLabel = (dim) => (dim.shortLabel ?? dim.label ?? dim.id).replace(/,\s*%$/, '');

/**
 * Step moves: every numeric parameter up and down, plus macro moves over the DCA grid.
 * Filter swaps are generated per round by {@link psChoiceMoves}.
 */
export const psBuildMoves = (space) => {
  const moves = [];
  for (const dim of space.dimensions) {
    if (psIsChoice(dim) || dim.kind === 'conditionAddInterval' || !(dim.max - dim.min > EPS)) {
      continue;
    }
    for (const sign of [1, -1]) {
      moves.push({
        id: `${dim.id}:${sign > 0 ? '+' : '-'}`,
        kind: 'step',
        label: `${shortLabel(dim)} ${sign > 0 ? '↑' : '↓'}`,
        group: dim.group,
        priority: stepPriority(dim),
        deltas: [{ dimId: dim.id, sign }],
        dimIds: [dim.id],
        sign,
        macro: false,
      });
    }
  }

  const byOrder = (kind) =>
    space.dimensions
      .filter((dim) => dim.kind === kind && typeof dim.orderIndex === 'number' && dim.max - dim.min > EPS)
      .sort((a, b) => a.orderIndex - b.orderIndex);
  const macro = (id, label, priority, group, deltas) =>
    moves.push({
      id,
      kind: 'step',
      label,
      group,
      priority,
      deltas,
      dimIds: deltas.map((delta) => delta.dimId),
      sign: 1,
      macro: true,
    });
  const halves = (dims) => {
    const half = Math.floor(dims.length / 2);
    return { near: dims.slice(0, half), far: dims.slice(dims.length - half) };
  };

  const indents = byOrder('orderIndent').filter((dim) => dim.orderIndex >= 1);
  if (indents.length >= 3) {
    const all = (sign) => indents.map((dim) => ({ dimId: dim.id, sign }));
    macro('macro:depth:+', 'Сетка: все отступы шире', 20, 'grid', all(1));
    macro('macro:depth:-', 'Сетка: все отступы уже', 20, 'grid', all(-1));
    const { near, far } = halves(indents);
    const shape = (sign) => [
      ...near.map((dim) => ({ dimId: dim.id, sign: -sign })),
      ...far.map((dim) => ({ dimId: dim.id, sign })),
    ];
    macro('macro:shape:+', 'Сетка: ближние ордера ближе, дальние дальше', 24, 'grid', shape(1));
    macro('macro:shape:-', 'Сетка: ровнее (ближние дальше, дальние ближе)', 24, 'grid', shape(-1));
  }

  const volumes = byOrder('orderVolume');
  if (volumes.length >= 3) {
    const { near, far } = halves(volumes);
    const shift = (sign) => [
      ...near.map((dim) => ({ dimId: dim.id, sign: -sign })),
      ...far.map((dim) => ({ dimId: dim.id, sign })),
    ];
    macro('macro:martingale:+', 'Объёмы: больше на дальние ордера', 22, 'grid', shift(1));
    macro('macro:martingale:-', 'Объёмы: больше на ближние ордера', 22, 'grid', shift(-1));
  }

  return moves.sort((a, b) => a.priority - b.priority || compareText(a.id, b.id));
};

/**
 * Filter swap moves around `genome`: revert to the original filter, switch it off, and
 * `samples` random alternatives (plus alternatives that improved before).
 */
export const psChoiceMoves = (space, genome, rng, samples, remembered = new Map()) => {
  const moves = [];
  for (const dim of space.dimensions) {
    if (!psIsChoice(dim) || dim.choices.length < 2) {
      continue;
    }
    const current = clamp(Math.round(psValue(genome, dim)), 0, dim.choices.length - 1);
    const picks = new Set();
    if (current !== 0) {
      picks.add(0);
    }
    const off = dim.choices.findIndex((choice) => choice.kind === 'off');
    if (off >= 0 && off !== current) {
      picks.add(off);
    }
    for (const index of remembered.get(dim.id) ?? []) {
      if (index !== current) {
        picks.add(index);
      }
    }
    const pool = [];
    dim.choices.forEach((choice, index) => {
      if (choice.kind === 'set' && index !== current && !picks.has(index)) {
        pool.push(index);
      }
    });
    for (let i = 0; i < samples && pool.length > 0; i++) {
      const at = Math.floor(rng() * pool.length);
      picks.add(pool.splice(at, 1)[0]);
    }
    for (const index of [...picks].sort((a, b) => a - b)) {
      const choice = dim.choices[index];
      moves.push({
        id: `${dim.id}=${index}`,
        kind: 'choice',
        label: `${shortLabel(dim)}: ${choice?.label ?? index}`,
        group: dim.group,
        priority: dim.kind === 'conditionAdd' ? 45 : 40,
        dimId: dim.id,
        dimIds: [dim.id],
        choice: index,
        macro: false,
      });
    }
  }
  return moves;
};

/** Moves a single dimension by `step` × its range (at least one grid step). */
const shiftValue = (dim, value, sign, step) => {
  const range = dim.max - dim.min;
  const delta = dim.integer ? Math.max(1, Math.round(step * range)) : Math.max(step * range, dim.step || 0);
  return clamp(value + sign * delta, dim.min, dim.max);
};

/** Applies a move; returns the repaired genome, or null when nothing changes (e.g. at a bound). */
export const psApplyMove = (space, genome, move, step, repair) => {
  const next = { ...genome };
  if (move.kind === 'choice') {
    next[move.dimId] = move.choice;
  } else {
    const dims = dimensionMap(space);
    for (const { dimId, sign } of move.deltas) {
      const dim = dims.get(dimId);
      if (dim) {
        next[dimId] = shiftValue(dim, psValue(genome, dim), sign, step);
      }
    }
  }
  const repaired = repair(next);
  return psSameGenome(space, repaired, genome) ? null : repaired;
};

/** Hooke–Jeeves pattern move: repeat the change from `from` to `to` once more. */
export const psPatternPoint = (space, from, to, repair) => {
  const next = { ...to };
  for (const dim of space.dimensions) {
    if (psIsChoice(dim)) {
      continue;
    }
    const delta = psValue(to, dim) - psValue(from, dim);
    if (Math.abs(delta) > EPS) {
      next[dim.id] = clamp(psValue(to, dim) + delta, dim.min, dim.max);
    }
  }
  const repaired = repair(next);
  return psSameGenome(space, repaired, to) ? null : repaired;
};

/**
 * Compares a variant with the original after choosing the bot deposit multiplier `scale`
 * (within [1/(1+maxChange), 1+maxChange]) that balances profit gain and drawdown gain.
 * Profit and drawdown of the whole portfolio are linear in the deposit, so the variant is
 * "better on both" exactly when the returned score is above 1.
 * Returns null when the original is not profitable (no meaningful reference).
 *
 * @param {{profit: number, risk: number}} candidate raw profit and drawdown (USDT)
 * @param {{profit: number, risk: number}} reference the original on the same coins and period
 * @param {number} maxChange allowed deposit change, e.g. 0.3 for ±30% (0 — keep the deposit)
 */
export const psRescale = (candidate, reference, maxChange) => {
  if (!(reference.profit > EPS && reference.risk > EPS)) {
    return null;
  }
  const limit = Math.max(0, Number.isFinite(maxChange) ? maxChange : 0);
  let scale = 1;
  if (limit > 0 && candidate.profit > EPS) {
    const ideal = Math.sqrt((reference.profit * reference.risk) / (candidate.profit * Math.max(candidate.risk, EPS)));
    scale = roundTo(clamp(ideal, 1 / (1 + limit), 1 + limit), 2);
  }
  const profitGain = (scale * candidate.profit) / reference.profit;
  const riskGain = Math.min(100, reference.risk / Math.max(scale * candidate.risk, EPS));
  return {
    scale,
    profitGain: roundTo(profitGain, 4),
    riskGain: roundTo(riskGain, 4),
    score: roundTo(Math.min(profitGain, riskGain), 4),
  };
};

/** Portfolio metrics that are proportional to the bot deposit. */
const DEPOSIT_LINEAR_METRICS = [
  'totalProfit',
  'activeMae',
  'adjustedProfit',
  'maxConcurrentMae',
  'maxRealizedDrawdown',
  'risk',
  'monthlyPercent',
  'drawdownPercent',
];

/** Portfolio metrics as they would be with the bot deposit multiplied by `scale`. */
export const psScaleMetrics = (metrics, scale) => {
  if (Math.abs(scale - 1) < EPS) {
    return metrics;
  }
  const out = { ...metrics };
  for (const key of DEPOSIT_LINEAR_METRICS) {
    if (typeof out[key] === 'number') {
      out[key] = roundTo(out[key] * scale, 4);
    }
  }
  return out;
};

/**
 * Score of one fold (time slice or coin group) against the original on the same fold,
 * with the variant's deposit multiplier applied. Mirrors the robustness report: a fold where
 * the original lost money is compared by the difference relative to its drawdown.
 */
export const psFoldScore = (candidate, base, scale, totalDeposit) => {
  const floorRisk = Math.max(totalDeposit, EPS) * 0.001;
  const floorProfit = Math.max(totalDeposit, EPS) * 0.0002;
  const profit = candidate.profit * scale;
  const risk = candidate.risk * scale;
  const profitTerm =
    base.profit > floorProfit ? profit / base.profit : 1 + (profit - base.profit) / Math.max(base.risk, floorRisk);
  const riskTerm = Math.max(base.risk, floorRisk) / Math.max(risk, floorRisk);
  return roundTo(Math.min(profitTerm, riskTerm), 4);
};

/** Search objective: overall score, capped by the median fold score (improvement must be broad). */
export const psObjective = (overall, foldScores) => {
  if (!Number.isFinite(overall) || !foldScores || foldScores.length < 2) {
    return overall;
  }
  return roundTo(Math.min(overall, psMedian(foldScores)), 4);
};

const monthLabel = (ms) => new Date(ms).toISOString().slice(0, 7);

/** Equal time slices (about half a year each, 2…8) of the search period; none if it is too short. */
export const psTimeFolds = (fromIso, toIso) => {
  const from = Date.parse(fromIso);
  const to = Date.parse(toIso);
  if (!(Number.isFinite(from) && Number.isFinite(to)) || (to - from) / DAY_MS < 300) {
    return [];
  }
  const count = clamp(Math.round((to - from) / DAY_MS / 182.625), 2, 8);
  const size = (to - from) / count;
  return Array.from({ length: count }, (_, index) => {
    const start = Math.round(from + index * size);
    const end = index === count - 1 ? to + 1 : Math.round(from + (index + 1) * size);
    return { key: `t${index}`, kind: 'time', label: `${monthLabel(start)} — ${monthLabel(end - 1)}`, from: start, to: end };
  });
};

/**
 * Splits the search coins into 2–3 groups with a similar mix of strong and weak coins
 * (snake order over `rank`, e.g. the original's profit per coin).
 */
export const psCoinGroups = (symbols, rank = null) => {
  const count = symbols.length >= 9 ? 3 : symbols.length >= 4 ? 2 : 0;
  if (count === 0) {
    return [];
  }
  const sorted = [...symbols].sort((a, b) => (rank?.get(b) ?? 0) - (rank?.get(a) ?? 0) || compareText(a, b));
  const groups = Array.from({ length: count }, () => []);
  sorted.forEach((symbol, index) => {
    const position = index % count;
    groups[Math.floor(index / count) % 2 === 0 ? position : count - 1 - position].push(symbol);
  });
  return groups.map((group, index) => ({
    key: `c${index}`,
    kind: 'coins',
    label: group.map((symbol) => symbol.split('/')[0]).join(', '),
    symbols: group,
  }));
};

/** Deals opened within [from, to) — same rule as the robustness report. */
export const psSliceEntries = (entries, from, to) => {
  const out = [];
  for (const entry of entries) {
    const start = Math.max(Date.parse(entry.from), from);
    const end = Math.min(Date.parse(entry.to), to);
    if (end > start) {
      out.push({
        ...entry,
        from: new Date(start).toISOString(),
        to: new Date(end).toISOString(),
        deals: entry.deals.filter((deal) => deal.s >= from && deal.s < to),
      });
    }
  }
  return out;
};

/** Multiplies deal results by the deposit multiplier (portfolio metrics are linear in it). */
export const psScaleEntries = (entries, scale) =>
  Math.abs(scale - 1) < EPS
    ? entries
    : entries.map((entry) => ({
        ...entry,
        depositAmount: entry.depositAmount * scale,
        deals: entry.deals.map((deal) => ({ ...deal, n: deal.n * scale, m: deal.m * scale })),
      }));

/**
 * Raw profit / drawdown of each fold. `metricsOf(entries)` is the portfolio aggregation
 * (blocking, dynamic limit…) returning `{profit, risk}`.
 */
export const psFoldMetrics = (entries, folds, metricsOf) =>
  folds.map((fold) => {
    const subset =
      fold.kind === 'time'
        ? psSliceEntries(entries, fold.from, fold.to)
        : entries.filter((entry) => fold.symbols.includes(entry.symbol));
    const metrics = subset.length > 0 ? metricsOf(subset) : { profit: 0, risk: 0 };
    return { key: fold.key, profit: roundTo(metrics.profit, 4), risk: roundTo(metrics.risk, 4) };
  });

/** Fold scores of a variant against the original's fold metrics. */
export const psFoldScores = (candidateFolds, baseFolds, scale, totalDeposit) => {
  const base = new Map(baseFolds.map((fold) => [fold.key, fold]));
  const out = [];
  for (const fold of candidateFolds) {
    const reference = base.get(fold.key);
    if (reference) {
      out.push({ key: fold.key, score: psFoldScore(fold, reference, scale, totalDeposit) });
    }
  }
  return out;
};

/**
 * Coins for the search: those that make the portfolio drawdown at its worst moment, a spread
 * across the profit ranking (strong, average and losing coins) and the most active ones.
 */
export const psPickScreeningSymbols = (entries, count) => {
  const limit = Math.max(1, Math.min(Math.round(count), entries.length));
  if (entries.length <= limit) {
    return entries.map((entry) => entry.symbol);
  }
  const events = [];
  for (const entry of entries) {
    for (const deal of entry.deals) {
      if (deal.st === 'C' || deal.e <= deal.s) {
        continue;
      }
      events.push({ t: deal.s, symbol: entry.symbol, mae: deal.m, open: true });
      events.push({ t: deal.e, symbol: entry.symbol, mae: deal.m, open: false });
    }
  }
  events.sort((a, b) => a.t - b.t || (a.open === b.open ? 0 : a.open ? 1 : -1));
  const open = new Map();
  let total = 0;
  let peak = -1;
  let atPeak = new Map();
  for (const event of events) {
    if (event.open) {
      open.set(event.symbol, (open.get(event.symbol) ?? 0) + event.mae);
      total += event.mae;
      if (total > peak) {
        peak = total;
        atPeak = new Map(open);
      }
    } else {
      const left = (open.get(event.symbol) ?? 0) - event.mae;
      if (left <= EPS) {
        open.delete(event.symbol);
      } else {
        open.set(event.symbol, left);
      }
      total -= event.mae;
    }
  }
  const stats = entries.map((entry) => {
    const finished = entry.deals.filter((deal) => deal.st === 'F');
    return {
      symbol: entry.symbol,
      atWorst: atPeak.get(entry.symbol) ?? 0,
      net: finished.reduce((sum, deal) => sum + deal.n, 0),
      deals: finished.length,
    };
  });
  const picked = [];
  const take = (list, quota, accept = () => true) => {
    let left = quota;
    for (const item of list) {
      if (picked.length >= limit || left <= 0) {
        return;
      }
      if (!picked.includes(item.symbol) && accept(item)) {
        picked.push(item.symbol);
        left -= 1;
      }
    }
  };
  const third = Math.ceil(limit / 3);
  take(
    [...stats].sort((a, b) => b.atWorst - a.atWorst || compareText(a.symbol, b.symbol)),
    third,
    (item) => item.atWorst > 0,
  );
  const rest = stats
    .filter((item) => !picked.includes(item.symbol))
    .sort((a, b) => b.net - a.net || compareText(a.symbol, b.symbol));
  if (rest.length > 0) {
    const spread = [];
    for (let i = 0; i < third; i++) {
      spread.push(rest[Math.min(rest.length - 1, Math.floor(((i + 0.5) * rest.length) / third))]);
    }
    take(spread, third);
  }
  take(
    [...stats].sort((a, b) => b.deals - a.deals || compareText(a.symbol, b.symbol)),
    limit,
  );
  const order = new Map(entries.map((entry, index) => [entry.symbol, index]));
  return picked.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
};

/**
 * Finalists for the check on all coins: the best variant, the simplest one that keeps at least
 * half of its gain (fewer changed parameters overfit less), then the next best ones.
 *
 * @param {{key: string, score: number, valid: boolean, changed: number, isBase?: boolean}[]} items
 */
export const psPickFinalists = (items, count, baseScore, margin = PS_DEFAULTS.margin) => {
  const limit = Math.max(1, Math.round(count));
  const valid = items
    .filter((item) => !item.isBase && item.valid)
    .sort((a, b) => b.score - a.score || a.changed - b.changed || compareText(a.key, b.key));
  const improving = valid.filter((item) => item.score > baseScore + margin);
  const pool = improving.length > 0 ? improving : valid;
  const picks = [];
  const add = (item) => {
    if (item && !picks.includes(item) && picks.length < limit) {
      picks.push(item);
    }
  };
  const best = pool[0];
  add(best);
  if (best && improving.length > 0) {
    const enough = baseScore + (best.score - baseScore) * 0.5;
    add(
      pool
        .filter((item) => item !== best && item.changed < best.changed && item.score >= enough)
        .sort((a, b) => a.changed - b.changed || b.score - a.score)[0],
    );
  }
  for (const item of pool) {
    add(item);
  }
  return { keys: picks.map((item) => item.key), improved: improving.length > 0 };
};

/** "TP 0.5→0.6, Подтяжка 0.4→0.3" — what differs between two genomes. */
export const psDescribeChanges = (space, from, to, format, limit = 4) => {
  const parts = [];
  for (const dim of space.dimensions) {
    const a = psValue(from, dim);
    const b = psValue(to, dim);
    if (Math.abs(a - b) > 1e-9) {
      parts.push(`${shortLabel(dim)} ${format(dim, a)}→${format(dim, b)}`);
    }
  }
  if (parts.length <= limit) {
    return parts.join(', ');
  }
  return `${parts.slice(0, limit).join(', ')} и ещё ${parts.length - limit}`;
};

/**
 * What a move changed: its own parameters first, then how many others were adjusted by the
 * repair (e.g. volumes re-balanced to keep the total).
 */
export const psDescribeMove = (space, from, to, move, format, limit = 3) => {
  const own = new Set(move.dimIds ?? []);
  const parts = [];
  let related = 0;
  for (const dim of space.dimensions) {
    const a = psValue(from, dim);
    const b = psValue(to, dim);
    if (Math.abs(a - b) <= 1e-9) {
      continue;
    }
    if (own.has(dim.id)) {
      parts.push(`${shortLabel(dim)} ${format(dim, a)}→${format(dim, b)}`);
    } else {
      related += 1;
    }
  }
  const shown = parts.slice(0, limit);
  if (parts.length > limit) {
    shown.push(`и ещё ${parts.length - limit}`);
  }
  if (related > 0) {
    shown.push(`пересчитано связанных: ${related}`);
  }
  return shown.join(', ');
};

/**
 * Runs the step-by-step search.
 *
 * `evaluate(genomes)` evaluates a batch (in parallel) and resolves to one result per genome:
 * `{score, valid, violation}` plus any extra data, or `null` when the genome was not evaluated
 * (search stopped). Already evaluated genomes should resolve from a cache, which makes an
 * interrupted search resumable: running it again replays the same path for free.
 *
 * @returns {Promise<{best, base, path, sensitivity, rounds, evaluations, stoppedBy}>}
 */
export const psRunSearch = async (context) => {
  const options = { ...PS_DEFAULTS, ...(context.options ?? {}) };
  const { space, repair, keyOf, evaluate } = context;
  const budget = Math.max(1, Math.round(context.budget));
  const rng = context.rng ?? psCreateRng(1);
  const shouldStop = () => (context.shouldStop ? context.shouldStop() : false);
  const emit = (event) => context.onEvent?.(event);

  const requested = new Set();
  const results = new Map();
  const budgetLeft = () => budget - requested.size;

  const evaluateMany = async (genomes) => {
    const keys = genomes.map((genome) => keyOf(genome));
    const batch = [];
    const inBatch = new Set();
    genomes.forEach((genome, index) => {
      const key = keys[index];
      if (results.has(key) || inBatch.has(key)) {
        return;
      }
      if (!requested.has(key)) {
        if (requested.size >= budget) {
          return;
        }
        requested.add(key);
      }
      inBatch.add(key);
      batch.push({ key, genome });
    });
    if (batch.length > 0) {
      const out = await evaluate(batch.map((item) => item.genome));
      out.forEach((result, index) => {
        if (result) {
          const { key, genome } = batch[index];
          results.set(key, { ...result, key, genome });
        }
      });
    }
    return keys.map((key) => results.get(key) ?? null);
  };

  const better = (a, b) => {
    if (!a) {
      return false;
    }
    if (!b) {
      return true;
    }
    if (a.valid !== b.valid) {
      return a.valid;
    }
    if (!a.valid) {
      return (a.violation ?? 0) < (b.violation ?? 0) - 1e-9;
    }
    return a.score > b.score + options.margin;
  };
  const delta = (result, current) =>
    result.valid && current.valid ? result.score - current.score : result.valid ? 1 : -1;

  const [base] = await evaluateMany([repair({ ...context.baseGenome })]);
  if (!base) {
    return { best: null, base: null, path: [], sensitivity: [], rounds: [], evaluations: requested.size, stoppedBy: 'stop' };
  }

  let current = base;
  const path = [{ key: base.key, genome: base.genome, score: base.score, round: 0, moves: [] }];
  const stepMoves = psBuildMoves(space);
  /** @type {Map<string, {tested: boolean, delta: number}>} */
  const moveState = new Map();
  const touchedDims = new Set();
  const rememberedChoices = new Map();
  const rounds = [];
  let sensitivity = [];
  let step = options.initialStep;
  let stall = 0;
  let round = 0;
  let stoppedBy = 'converged';

  const planned = (move) => {
    const state = moveState.get(move.id);
    if (!state?.tested) {
      return true;
    }
    if (move.dimIds.some((dimId) => touchedDims.has(dimId))) {
      return true;
    }
    return Math.abs(state.delta) >= options.insensitive && state.delta > -options.harmful;
  };

  while (true) {
    if (shouldStop()) {
      stoppedBy = 'stop';
      break;
    }
    if (budgetLeft() <= 0) {
      stoppedBy = 'budget';
      break;
    }
    if (step < options.minStep - 1e-12) {
      break;
    }
    round += 1;
    const roundStart = current;
    const before = requested.size;

    // 1. Scan.
    const moves = [
      ...stepMoves.filter(planned),
      ...psChoiceMoves(space, current.genome, rng, round === 1 ? options.choiceSamples : 0, rememberedChoices),
    ].sort((a, b) => a.priority - b.priority || compareText(a.id, b.id));
    const reserve = round === 1 ? Math.max(4, Math.ceil(budget * options.reserveShare)) : 2;
    let room = Math.max(0, budgetLeft() - reserve);
    const candidates = [];
    const seen = new Set([current.key]);
    for (const move of moves) {
      const genome = psApplyMove(space, current.genome, move, step, repair);
      if (!genome) {
        continue;
      }
      const key = keyOf(genome);
      if (seen.has(key)) {
        continue;
      }
      const fresh = !results.has(key) && !requested.has(key);
      if (fresh && room <= 0) {
        continue;
      }
      if (fresh) {
        room -= 1;
      }
      seen.add(key);
      candidates.push({ move, genome, key });
    }
    if (candidates.length === 0 && budgetLeft() <= reserve) {
      stoppedBy = 'budget';
      break;
    }
    emit({ type: 'scan', round, step, count: candidates.length, current });
    const scanned = await evaluateMany(candidates.map((candidate) => candidate.genome));
    const scan = [];
    candidates.forEach((candidate, index) => {
      const result = scanned[index];
      if (!result) {
        return;
      }
      const change = delta(result, current);
      scan.push({ ...candidate, result, delta: change });
      if (candidate.move.kind === 'step') {
        moveState.set(candidate.move.id, { tested: true, delta: change });
      } else if (better(result, current)) {
        const list = rememberedChoices.get(candidate.move.dimId) ?? [];
        rememberedChoices.set(candidate.move.dimId, [...new Set([...list, candidate.move.choice])]);
      }
    });
    if (round === 1) {
      sensitivity = scan.map((item) => ({ move: item.move, genome: item.genome, result: item.result, delta: item.delta }));
    }
    emit({ type: 'scanned', round, step, scan, current });
    if (shouldStop()) {
      stoppedBy = 'stop';
      break;
    }

    // 2. Climb: combine improving moves, then follow each accepted direction.
    const improving = scan
      .filter((item) => better(item.result, current))
      .sort((a, b) => b.result.score - a.result.score || compareText(a.move.id, b.move.id));
    const chosen = [];
    for (const item of improving) {
      const clash = chosen.some(
        (other) =>
          (item.move.kind === 'choice' && other.move.kind === 'choice' && other.move.dimId === item.move.dimId) ||
          (item.move.kind === 'step' &&
            other.move.kind === 'step' &&
            other.move.id.slice(0, -1) === item.move.id.slice(0, -1)),
      );
      if (!clash) {
        chosen.push(item);
      }
    }
    let accepted = [];
    if (chosen.length >= 2 && budgetLeft() > 0) {
      let genome = current.genome;
      for (const item of chosen) {
        genome = psApplyMove(space, genome, item.move, step, repair) ?? genome;
      }
      const [combined] = await evaluateMany([genome]);
      if (combined && better(combined, chosen[0].result)) {
        current = combined;
        accepted = chosen.map((item) => item.move);
        emit({ type: 'accept', how: 'combined', round, moves: accepted, result: combined });
      } else if (combined) {
        emit({ type: 'reject', how: 'combined', round, moves: chosen.map((item) => item.move), result: combined });
      }
    }
    if (accepted.length === 0) {
      for (const item of chosen) {
        if (shouldStop() || budgetLeft() <= 0) {
          break;
        }
        const genome = psApplyMove(space, current.genome, item.move, step, repair);
        if (!genome) {
          continue;
        }
        const [result] = await evaluateMany([genome]);
        if (!result) {
          break;
        }
        if (better(result, current)) {
          current = result;
          accepted.push(item.move);
          emit({ type: 'accept', how: 'greedy', round, moves: [item.move], result });
        } else {
          emit({ type: 'reject', how: 'greedy', round, moves: [item.move], result });
        }
      }
    }
    for (const move of accepted) {
      if (move.kind !== 'step') {
        continue;
      }
      for (let i = 0; i < options.maxLineSteps; i++) {
        if (shouldStop() || budgetLeft() <= 0) {
          break;
        }
        const genome = psApplyMove(space, current.genome, move, step, repair);
        if (!genome) {
          emit({ type: 'bound', round, move });
          break;
        }
        const [result] = await evaluateMany([genome]);
        if (!(result && better(result, current))) {
          break;
        }
        current = result;
        emit({ type: 'accept', how: 'line', round, moves: [move], result });
      }
    }

    // 3. Pattern move.
    if (current.key !== roundStart.key && !shouldStop() && budgetLeft() > 0) {
      const genome = psPatternPoint(space, roundStart.genome, current.genome, repair);
      if (genome) {
        const [result] = await evaluateMany([genome]);
        if (result && better(result, current)) {
          current = result;
          emit({ type: 'accept', how: 'pattern', round, moves: [], result });
        }
      }
    }

    const improved = current.key !== roundStart.key;
    if (improved) {
      for (const move of accepted) {
        for (const dimId of move.dimIds) {
          touchedDims.add(dimId);
        }
      }
      path.push({ key: current.key, genome: current.genome, score: current.score, round, moves: accepted });
    }
    const scores = scan.filter((item) => item.result.valid).map((item) => item.result.score);
    const info = {
      round,
      step,
      improved,
      best: current,
      start: roundStart,
      evaluated: requested.size - before,
      scanned: scan.length,
      medianScore: scores.length > 0 ? psMedian(scores) : current.score,
      validCount: scores.length,
      accepted,
    };
    rounds.push({
      round,
      step,
      improved,
      bestScore: current.score,
      medianScore: info.medianScore,
      evaluated: info.evaluated,
      validCount: info.validCount,
    });
    context.onRound?.(info);
    if (improved) {
      stall = 0;
    } else {
      stall += 1;
      step /= 2;
      if (stall >= options.stallRounds) {
        break;
      }
    }
  }
  if (stoppedBy === 'converged' && budgetLeft() <= 0) {
    stoppedBy = 'budget';
  }
  return { best: current, base, path, sensitivity, rounds, evaluations: requested.size, stoppedBy };
};

// ---------------------------------------------------------------------------------------------
// Exhaustive search over the discretised space ("полный перебор") and a coordinate-wise
// optimality check of the winner ("сертификат"). Both use the same `evaluate` contract as
// psRunSearch, so the engine's archive and the backtest cache make them resumable for free.

/** Number of decimals that represent multiples of `step` exactly. */
const stepDigits = (step) => (step > 0 ? Math.min(10, Math.max(0, 3 - Math.floor(Math.log10(step)))) : 6);

/** Every allowed value of one dimension: min..max by its step (indices for filter swaps). */
export const psGridValues = (dim) => {
  if (psIsChoice(dim)) {
    return dim.choices.map((_, index) => index);
  }
  const { min, max } = dim;
  if (!(max - min > EPS)) {
    return [min];
  }
  const step = dim.integer ? Math.max(1, Math.round(dim.step || 1)) : dim.step;
  if (!(step > 0)) {
    return [min, max];
  }
  const digits = stepDigits(step);
  const count = Math.floor((max - min) / step + 1e-6) + 1;
  const values = [];
  for (let i = 0; i < count; i++) {
    values.push(roundTo(min + i * step, digits));
  }
  if (max - values[values.length - 1] > step * 1e-6) {
    values.push(roundTo(max, digits));
  }
  return values;
};

/** Size of the full grid: `total` combinations (Infinity when it does not fit a double) and per-dimension counts. */
export const psGridSize = (space) => {
  const counts = space.dimensions.map((dim) => psGridValues(dim).length);
  const log10 = counts.reduce((sum, count) => sum + Math.log10(count), 0);
  const total = counts.reduce((product, count) => product * count, 1);
  return { total: Number.isFinite(total) ? total : Number.POSITIVE_INFINITY, log10, counts };
};

/** Variants checked in one pass of {@link psCertify}: every other value of every parameter. */
export const psCertifyCost = (space) =>
  space.dimensions.reduce((sum, dim) => sum + Math.max(0, psGridValues(dim).length - 1), 0);

const gcd = (a, b) => {
  let x = a;
  let y = b;
  while (y) {
    [x, y] = [y, x % y];
  }
  return x;
};

/**
 * Stride of the visiting order: k → k·stride mod total is a permutation of the grid, so a stopped
 * enumeration has sampled the whole space evenly instead of varying only the last parameters.
 */
export const psGridStride = (total) => {
  if (total <= 2) {
    return 1;
  }
  let stride = Math.max(1, Math.round(total * 0.6180339887));
  while (gcd(stride, total) !== 1) {
    stride += 1;
  }
  return stride;
};

/** Genome number `index` of the grid (mixed radix over `lists`, the first dimension changing fastest). */
const gridGenome = (space, lists, baseGenome, index) => {
  const genome = { ...baseGenome };
  let rest = index;
  space.dimensions.forEach((dim, position) => {
    const values = lists[position];
    genome[dim.id] = values[rest % values.length];
    rest = Math.floor(rest / values.length);
  });
  return genome;
};

const strictlyBetter = (a, b) => {
  if (!a) {
    return false;
  }
  if (!b) {
    return true;
  }
  if (a.valid !== b.valid) {
    return a.valid;
  }
  if (!a.valid) {
    return (a.violation ?? 0) < (b.violation ?? 0) - 1e-12;
  }
  return a.score > b.score + 1e-12;
};

/**
 * Evaluates every combination of the grid (after `repair`, equal genomes are evaluated once) and
 * returns the best. The guarantee holds when `stoppedBy === 'complete'`: no other combination of
 * the allowed values scored higher on the evaluated coins and period.
 *
 * `budget` is the largest grid the caller accepts; a bigger grid is not started ('too-large').
 *
 * @returns {Promise<{best, base, path, sensitivity, rounds, evaluations, stoppedBy, grid}>}
 */
export const psRunGrid = async (context) => {
  const { space, repair, keyOf, evaluate } = context;
  const shouldStop = () => (context.shouldStop ? context.shouldStop() : false);
  const emit = (event) => context.onEvent?.(event);
  const batchSize = Math.max(1, Math.round(context.batch ?? 24));
  const lists = space.dimensions.map((dim) => psGridValues(dim));
  const { total } = psGridSize(space);
  const grid = { total, visited: 0, unique: 0, evaluated: 0 };
  const point = (result) => ({ key: result.key, genome: result.genome, score: result.score, round: 0, moves: [] });

  const baseGenome = repair({ ...context.baseGenome });
  const baseKey = keyOf(baseGenome);
  const [baseResult] = await evaluate([baseGenome]);
  if (!baseResult) {
    return { best: null, base: null, path: [], sensitivity: [], rounds: [], evaluations: 0, stoppedBy: 'stop', grid };
  }
  const base = { ...baseResult, key: baseKey, genome: baseGenome };
  const seen = new Set([baseKey]);
  let best = base;
  let evaluated = 1;
  let stoppedBy = 'complete';
  const limit = Number.isFinite(context.budget) ? context.budget : Number.POSITIVE_INFINITY;
  if (!(total <= limit)) {
    stoppedBy = 'too-large';
  } else {
    const stride = psGridStride(total);
    for (let start = 0; start < total; start += batchSize) {
      if (shouldStop()) {
        stoppedBy = 'stop';
        break;
      }
      const end = Math.min(total, start + batchSize);
      const batch = [];
      for (let k = start; k < end; k++) {
        const genome = repair(gridGenome(space, lists, baseGenome, (k * stride) % total));
        const key = keyOf(genome);
        if (!seen.has(key)) {
          seen.add(key);
          batch.push({ genome, key });
        }
      }
      let complete = true;
      if (batch.length > 0) {
        const out = await evaluate(batch.map((item) => item.genome));
        out.forEach((result, index) => {
          if (!result) {
            complete = false;
            return;
          }
          evaluated += 1;
          const item = { ...result, key: batch[index].key, genome: batch[index].genome };
          if (strictlyBetter(item, best)) {
            best = item;
          }
        });
      }
      grid.visited = complete ? end : start;
      grid.unique = seen.size;
      grid.evaluated = evaluated;
      emit({ type: 'grid', visited: grid.visited, total, unique: seen.size, evaluated, best });
      if (!complete) {
        stoppedBy = 'stop';
        break;
      }
    }
  }
  grid.unique = seen.size;
  grid.evaluated = evaluated;
  return {
    best,
    base,
    path: best.key === base.key ? [point(base)] : [point(base), { ...point(best), round: 1 }],
    sensitivity: [],
    rounds: [],
    evaluations: evaluated,
    stoppedBy,
    grid,
  };
};

/**
 * Coordinate-wise check of `start`: every other value of every parameter is tried with the rest
 * fixed; an improvement (by more than `margin`) becomes the new point and the pass is repeated.
 * `certified` means a whole pass found no single-parameter change that improves the score.
 *
 * @returns {Promise<{best, certified, passes, evaluations, checked, moves, stoppedBy}>}
 */
export const psCertify = async (context) => {
  const { space, repair, keyOf, evaluate } = context;
  const shouldStop = () => (context.shouldStop ? context.shouldStop() : false);
  const emit = (event) => context.onEvent?.(event);
  const margin = context.margin ?? PS_DEFAULTS.margin;
  const maxPasses = Math.max(1, Math.round(context.maxPasses ?? 3));
  const budget = Number.isFinite(context.budget) ? Math.max(1, Math.round(context.budget)) : Number.POSITIVE_INFINITY;
  const better = (a, b) => {
    if (!a || !b || a.valid !== b.valid || !a.valid) {
      return strictlyBetter(a, b);
    }
    return a.score > b.score + margin;
  };
  const dims = space.dimensions
    .filter((dim) => psGridValues(dim).length > 1)
    .sort((a, b) => stepPriority(a) - stepPriority(b) || compareText(a.id, b.id));
  const requested = new Set();
  let current = context.start;
  let passes = 0;
  let checked = 0;
  let stoppedBy = 'passes';
  const moves = [];
  passLoop: while (passes < maxPasses) {
    passes += 1;
    let moved = false;
    checked = 0;
    for (const dim of dims) {
      if (shouldStop()) {
        stoppedBy = 'stop';
        break passLoop;
      }
      const from = psValue(current.genome, dim);
      const keys = new Set([current.key]);
      const batch = [];
      for (const value of psGridValues(dim)) {
        if (Math.abs(value - from) < 1e-9) {
          continue;
        }
        const genome = repair({ ...current.genome, [dim.id]: value });
        const key = keyOf(genome);
        if (!keys.has(key)) {
          keys.add(key);
          batch.push({ genome, key });
        }
      }
      if (batch.length === 0) {
        continue;
      }
      const fresh = batch.filter((item) => !requested.has(item.key)).length;
      if (requested.size + fresh > budget) {
        stoppedBy = 'budget';
        break passLoop;
      }
      for (const item of batch) {
        requested.add(item.key);
      }
      const out = await evaluate(batch.map((item) => item.genome));
      if (out.some((result) => !result)) {
        stoppedBy = 'stop';
        break passLoop;
      }
      checked += batch.length;
      let bestHere = null;
      out.forEach((result, index) => {
        const item = { ...result, key: batch[index].key, genome: batch[index].genome };
        if (strictlyBetter(item, bestHere)) {
          bestHere = item;
        }
      });
      emit({ type: 'certify-dim', pass: passes, dim, checked: batch.length, best: bestHere, current });
      if (better(bestHere, current)) {
        const move = { pass: passes, dimId: dim.id, label: shortLabel(dim), from, to: psValue(bestHere.genome, dim), score: bestHere.score };
        moves.push(move);
        emit({ type: 'certify-move', ...move, result: bestHere, previous: current });
        current = bestHere;
        moved = true;
      }
    }
    emit({ type: 'certify-pass', pass: passes, moved, checked });
    if (!moved) {
      return { best: current, certified: true, passes, evaluations: requested.size, checked, moves, stoppedBy: 'certified' };
    }
  }
  return { best: current, certified: false, passes, evaluations: requested.size, checked, moves, stoppedBy };
};
