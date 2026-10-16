// Optimizer modes (Veles Tools 2.16): besides "improve my strategy" the optimizer can search for a
// scalper, a SHORT bot, a LONG DCA bot from a template and the best set of coins for a bot.
//
// The template modes do not invent Veles enums: a template is built on top of the user's bot (the
// "shell"), which keeps its exchange, deposit, margin type, settings.type/indentType, take-profit type
// and stop-loss shape; only the grid, the entry filters, the take-profit value and the stop-loss
// indent are generated. The search itself is the 2.15 engine (pattern search over the template's
// parameters, filter swaps from a mode-specific indicator pool), scored by an absolute objective
// (stable profit-to-drawdown) instead of "better than the original".
//
// Everything here is pure and has no dependency on the app bundle (bundle helpers are injected).

const EPS = 1e-9;
const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;
const MONTH_DAYS = 30.4375;

/** @param {number} value @param {number} min @param {number} max */

/** @param {number} value @param {number} digits */
const roundTo = (value, digits) => {
  const factor = 10 ** digits;
  const rounded = Math.round(value * factor) / factor;
  return rounded === 0 ? 0 : rounded;
};

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const clone = (value) => (value === undefined ? undefined : JSON.parse(JSON.stringify(value)));

/** Median of finite numbers (null for an empty list). */
export const vmMedian = (values) => {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (sorted.length === 0) {
    return null;
  }
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

// ═══════════════════════════════════════════════════════════════════
// MODES
// ═══════════════════════════════════════════════════════════════════

/** Score metric of the template modes: profit-to-drawdown, discounted when it is not stable. */
export const VM_STABLE_METRIC = 'vtStablePnlToRisk';

export const VM_GOALS = Object.freeze([
  {
    id: 'improve',
    tab: '🎯 Улучшить стратегию',
    title: '🎯 Найти устойчивые настройки',
    button: 'Найти устойчивые настройки',
    template: false,
  },
  {
    id: 'scalper',
    tab: '⚡ Скальпер',
    title: '⚡ Найти лучший скальпер',
    button: 'Найти скальпер',
    template: true,
  },
  {
    id: 'short',
    tab: '📉 Шорт-бот',
    title: '📉 Найти лучший шорт-бот',
    button: 'Найти шорт-бот',
    template: true,
  },
  {
    id: 'long',
    tab: '📈 LONG DCA с нуля',
    title: '📈 Собрать LONG DCA-бота с нуля',
    button: 'Собрать LONG-бота',
    template: true,
  },
  {
    id: 'coins',
    tab: '🪙 Подбор монет',
    title: '🪙 Подобрать монеты для бота',
    button: 'Подобрать монеты',
    template: false,
  },
]);

export const VM_GOAL_IDS = Object.freeze(VM_GOALS.map((goal) => goal.id));

/** @param {string | null | undefined} goal */
export const vmGoal = (goal) => VM_GOALS.find((item) => item.id === goal) ?? VM_GOALS[0];

/** Mode whose base strategy is generated from a template (not the user's bot as is). */
export const vmIsTemplateGoal = (goal) => vmGoal(goal).template;

/** Mode scored by an absolute objective (template modes and coin selection). */
export const vmIsAbsoluteGoal = (goal) => goal === 'scalper' || goal === 'short' || goal === 'long' || goal === 'coins';

/** Template profiles per mode (shown in the mode card). */
export const VM_PROFILES = Object.freeze({
  scalper: [
    { id: 'grid2', label: 'Вход + 2 усреднения', hint: 'база 40% и два близких усреднения, тейк ~0.5%' },
    { id: 'single', label: 'Один ордер', hint: 'весь объём сразу, быстрый тейк и стоп' },
    { id: 'grid4', label: 'Мини-сетка (4 усреднения)', hint: 'сетка до ~4% с небольшим мартингейлом' },
  ],
  long: [
    { id: 'classic', label: 'Классическая сетка', hint: '8 ордеров до ~18%, тейк ~1%' },
    { id: 'tight', label: 'Частая плотная сетка', hint: '10 ордеров до ~10%, тейк ~0.7%' },
    { id: 'wide', label: 'Широкая сетка', hint: '7 ордеров до ~30%, тейк ~1.5%, для глубоких просадок' },
  ],
  short: [
    { id: 'mirror', label: 'Зеркало моего LONG-бота', hint: 'условия и сетка бота, отражённые для SHORT' },
    { id: 'classic', label: 'SHORT DCA-шаблон', hint: '8 ордеров вверх до ~18%, вход по перекупленности' },
    { id: 'tight', label: 'SHORT плотная сетка', hint: '10 ордеров до ~10%, тейк ~0.7%' },
  ],
});

export const VM_TIMEFRAMES = Object.freeze([
  { id: 'ONE_MINUTE', label: '1 мин' },
  { id: 'FIVE_MINUTES', label: '5 мин' },
  { id: 'FIFTEEN_MINUTES', label: '15 мин' },
]);

/** Default template options of a mode. */
export const vmDefaultTemplate = (goal) => {
  switch (goal) {
    case 'scalper':
      return { direction: 'LONG', timeframe: 'FIVE_MINUTES', profile: 'grid2', stopLoss: true };
    case 'short':
      return { direction: 'SHORT', timeframe: 'ONE_HOUR', profile: 'mirror', stopLoss: false };
    case 'long':
      return { direction: 'LONG', timeframe: 'ONE_HOUR', profile: 'classic', stopLoss: false };
    default:
      return null;
  }
};

/** Template options with defaults filled in and invalid values replaced. */
export const vmNormalizeTemplate = (goal, template) => {
  const defaults = vmDefaultTemplate(goal);
  if (!defaults) {
    return null;
  }
  const value = isRecord(template) ? template : {};
  const profiles = VM_PROFILES[goal] ?? [];
  const profile = profiles.some((item) => item.id === value.profile) ? value.profile : defaults.profile;
  const direction =
    goal === 'short' ? 'SHORT' : goal === 'long' ? 'LONG' : value.direction === 'SHORT' ? 'SHORT' : 'LONG';
  const timeframe =
    goal === 'scalper' && VM_TIMEFRAMES.some((item) => item.id === value.timeframe) ? value.timeframe : defaults.timeframe;
  return {
    direction,
    timeframe,
    profile,
    stopLoss: typeof value.stopLoss === 'boolean' ? value.stopLoss : defaults.stopLoss,
  };
};

// ═══════════════════════════════════════════════════════════════════
// SHORT MIRROR
// ═══════════════════════════════════════════════════════════════════

/** Directional value domains: how a threshold is mirrored between LONG and SHORT. */
const MIRROR_CENTER = Object.freeze({
  RSI: 50,
  MFI: 50,
  STOCHASTIC: 50,
  CONNORS_RSI: 50,
  RVI: 50,
  CMO: 0,
  CCI: 0,
  WILLIAMS_R: -50,
  ROC: 0,
  MOMENTUM: 0,
  PRICE_CHANGE: 0,
  PRICE_CHANGE_PERIOD: 0,
});

const OPPOSITE_OPERATION = Object.freeze({
  LESS: 'GREATER',
  GREATER: 'LESS',
  LESS_OR_EQUAL: 'GREATER_OR_EQUAL',
  GREATER_OR_EQUAL: 'LESS_OR_EQUAL',
  CROSS_UP: 'CROSS_DOWN',
  CROSS_DOWN: 'CROSS_UP',
});

/** Indicator family key: "PRICE_CHANGE_PERIOD_10" → "PRICE_CHANGE_PERIOD", "RSI" → "RSI". */
export const vmIndicatorBase = (indicator) =>
  typeof indicator === 'string' ? indicator.replace(/_\d+$/, '') : '';

/** Mirrored threshold of a directional indicator (null when the indicator is not directional). */
export const vmMirrorValue = (indicator, value) => {
  const center = MIRROR_CENTER[vmIndicatorBase(indicator)];
  if (center === undefined || !Number.isFinite(value)) {
    return null;
  }
  return roundTo(2 * center - value, 6);
};

/**
 * Legacy condition mirrored for SHORT. Threshold conditions of directional indicators get the
 * opposite operation and a mirrored value; non-directional ones (ADX, ATR, volume…) and signal
 * indicators (`basic`) are kept. Returns `{condition, changed, signal}`.
 */
export const vmMirrorCondition = (condition) => {
  if (!isRecord(condition)) {
    return { condition, changed: false, signal: false };
  }
  const out = { ...condition };
  if (condition.basic === true || condition.value === null || condition.value === undefined) {
    return { condition: out, changed: false, signal: typeof condition.indicator === 'string' };
  }
  const mirrored = vmMirrorValue(condition.indicator, Number(condition.value));
  const operation = OPPOSITE_OPERATION[condition.operation];
  if (mirrored === null || !operation) {
    return { condition: out, changed: false, signal: false };
  }
  out.value = mirrored;
  out.operation = operation;
  return { condition: out, changed: true, signal: false };
};

/** Tree-format node (conditionGroups) mirrored for SHORT: `INDICATOR op CONSTANT` comparisons. */
const mirrorTreeNode = (node, notes) => {
  if (!isRecord(node) || node.type !== 'OPERATION') {
    if (isRecord(node) && node.type === 'SIGNAL') {
      notes.signals += 1;
    }
    return node;
  }
  const out = { ...node };
  const left = node.left;
  const right = node.right;
  const operation = OPPOSITE_OPERATION[node.operator];
  if (isRecord(left) && left.type === 'INDICATOR' && isRecord(right) && right.type === 'CONSTANT' && operation) {
    const mirrored = vmMirrorValue(left.indicator, Number(right.value));
    if (mirrored !== null) {
      out.operator = operation;
      out.right = { ...right, value: mirrored };
      notes.changed += 1;
    }
  } else if (isRecord(right) && right.type === 'INDICATOR' && isRecord(left) && left.type === 'CONSTANT' && operation) {
    const mirrored = vmMirrorValue(right.indicator, Number(left.value));
    if (mirrored !== null) {
      out.operator = operation;
      out.left = { ...left, value: mirrored };
      notes.changed += 1;
    }
  }
  return out;
};

const mirrorConditionHolder = (holder, notes) => {
  if (!isRecord(holder)) {
    return holder;
  }
  const out = { ...holder };
  if (Array.isArray(holder.conditions)) {
    out.conditions = holder.conditions.map((condition) => {
      const result = vmMirrorCondition(condition);
      notes.changed += result.changed ? 1 : 0;
      notes.signals += result.signal ? 1 : 0;
      return result.condition;
    });
  }
  if (Array.isArray(holder.conditionGroups)) {
    out.conditionGroups = holder.conditionGroups.map((group) =>
      Array.isArray(group) ? group.map((node) => mirrorTreeNode(node, notes)) : group,
    );
  }
  return out;
};

/**
 * The bot mirrored for SHORT: `algorithm: "SHORT"` and every threshold condition (entry, grid
 * orders, take-profit, stop-loss) mirrored. Grid indents, volumes, take-profit and stop-loss
 * percentages are direction-free in Veles and stay as they are. A bot that already is SHORT is
 * returned unchanged. `notes` counts mirrored conditions and kept signal indicators.
 */
export const vmMirrorToShort = (strategy) => {
  const notes = { changed: 0, signals: 0, alreadyShort: false };
  if (!isRecord(strategy)) {
    return { strategy, notes };
  }
  if (strategy.algorithm === 'SHORT') {
    notes.alreadyShort = true;
    return { strategy: clone(strategy), notes };
  }
  const out = mirrorConditionHolder(clone(strategy), notes);
  out.algorithm = 'SHORT';
  if (isRecord(out.settings)) {
    const settings = { ...out.settings };
    settings.baseOrder = mirrorConditionHolder(settings.baseOrder, notes);
    if (Array.isArray(settings.orders)) {
      settings.orders = settings.orders.map((order) => mirrorConditionHolder(order, notes));
    }
    out.settings = settings;
  }
  out.profit = mirrorConditionHolder(out.profit, notes);
  out.stopLoss = mirrorConditionHolder(out.stopLoss, notes);
  return { strategy: out, notes };
};

// ═══════════════════════════════════════════════════════════════════
// TEMPLATES
// ═══════════════════════════════════════════════════════════════════

/** Legacy threshold condition as Veles stores it. */
export const vmCondition = (indicator, interval, operation, value) => ({
  type: 'INDICATOR',
  indicator,
  interval,
  basic: false,
  value,
  operation,
  closed: true,
  reverse: false,
});

/**
 * Grid orders: cumulative indents from the entry (`count` orders including the base order at 0)
 * growing with `power`, and volumes growing by `ratio`, normalised to 100 % in whole percent.
 * For `indentType: "PREVIOUS"` the indents are written as steps from the previous order.
 */
export const vmGrid = ({ count, lastIndent, power = 1.2, ratio = 1.3, firstIndent = null }, indentType = null) => {
  const total = Math.max(1, Math.round(count));
  const raw = Array.from({ length: total }, (_, index) => ratio ** index);
  const sum = raw.reduce((acc, value) => acc + value, 0);
  const volumes = raw.map((value) => Math.max(1, Math.round((value / sum) * 100)));
  const drift = 100 - volumes.reduce((acc, value) => acc + value, 0);
  volumes[volumes.length - 1] = Math.max(1, volumes[volumes.length - 1] + drift);
  const indents = [0];
  for (let index = 1; index < total; index += 1) {
    const share = index / (total - 1);
    const value = lastIndent * share ** power;
    indents.push(roundTo(index === 1 && firstIndent !== null ? firstIndent : value, 2));
  }
  const written =
    indentType === 'PREVIOUS' ? indents.map((value, index) => (index === 0 ? 0 : roundTo(value - indents[index - 1], 2))) : indents;
  const orders = written.map((indent, index) => ({ indent, volume: volumes[index], conditions: [] }));
  return { baseOrder: orders[0], orders: orders.slice(1) };
};

/** Explicit grid (indents from the entry, volumes in %) in the bot's indent convention. */
const explicitGrid = (indents, volumes, indentType) => {
  const written =
    indentType === 'PREVIOUS' ? indents.map((value, index) => (index === 0 ? 0 : roundTo(value - indents[index - 1], 2))) : indents;
  const orders = written.map((indent, index) => ({ indent, volume: volumes[index], conditions: [] }));
  return { baseOrder: orders[0], orders: orders.slice(1) };
};

const PROFILE_SPECS = Object.freeze({
  scalper: {
    grid2: { grid: { indents: [0, 0.8, 1.8], volumes: [40, 30, 30] }, takeProfit: 0.5, stopLoss: 3 },
    single: { grid: { indents: [0], volumes: [100] }, takeProfit: 0.5, stopLoss: 1.5 },
    grid4: { grid: { count: 5, lastIndent: 4, power: 1.15, ratio: 1.25, firstIndent: 0.6 }, takeProfit: 0.6, stopLoss: 6 },
  },
  long: {
    classic: { grid: { count: 8, lastIndent: 18, power: 1.25, ratio: 1.3 }, takeProfit: 1, pullUp: 0.3 },
    tight: { grid: { count: 10, lastIndent: 10, power: 1.15, ratio: 1.2 }, takeProfit: 0.7, pullUp: 0.2 },
    wide: { grid: { count: 7, lastIndent: 30, power: 1.4, ratio: 1.45 }, takeProfit: 1.5, pullUp: 0.4 },
  },
});

/** Entry and take-profit-signal filters of a template, LONG semantics (mirrored for SHORT). */
const templateFilters = (goal, profile, timeframe) => {
  if (goal === 'scalper') {
    return {
      entry: [vmCondition('RSI', timeframe, 'LESS', 30), vmCondition('CCI', timeframe, 'LESS', -100)],
      exit: [vmCondition('RSI', timeframe, 'GREATER', 55)],
    };
  }
  if (profile === 'tight') {
    return {
      entry: [vmCondition('RSI', 'FIFTEEN_MINUTES', 'LESS', 30), vmCondition('CCI', 'FIFTEEN_MINUTES', 'LESS', -120)],
      exit: [vmCondition('RSI', 'FIFTEEN_MINUTES', 'GREATER', 60)],
    };
  }
  if (profile === 'wide') {
    return {
      entry: [vmCondition('RSI', 'ONE_HOUR', 'LESS', 30), vmCondition('MFI', 'ONE_HOUR', 'LESS', 25)],
      exit: [vmCondition('RSI', 'ONE_HOUR', 'GREATER', 65)],
    };
  }
  return {
    entry: [vmCondition('RSI', 'ONE_HOUR', 'LESS', 35), vmCondition('CCI', 'FIFTEEN_MINUTES', 'LESS', -100)],
    exit: [vmCondition('RSI', 'ONE_HOUR', 'GREATER', 65)],
  };
};

/** Quote currency of a bot (pair.to, or the part after "/" of its first symbol). */
const quoteOf = (strategy) => {
  if (isRecord(strategy?.pair) && typeof strategy.pair.to === 'string' && strategy.pair.to) {
    return strategy.pair.to.toUpperCase();
  }
  const symbol = Array.isArray(strategy?.symbols) ? strategy.symbols[0] : strategy?.symbol;
  if (typeof symbol === 'string' && symbol.includes('/')) {
    return symbol.split('/')[1].trim().toUpperCase() || 'USDT';
  }
  return 'USDT';
};

/** Coins written into the template (`symbols`), so «Создать бота» offers the run's coins. */
const templateSymbols = (symbols, quote) =>
  (Array.isArray(symbols) ? symbols : [])
    .map((symbol) => String(symbol).trim().toUpperCase())
    .filter(Boolean)
    .map((symbol) => (symbol.includes('/') ? symbol : `${symbol}/${quote}`));

/**
 * Template strategy of a mode, built on the user's bot `shell`. Keeps every field of the shell
 * the engine does not search (exchange, pair type, deposit, margin, settings.type/indentType,
 * take-profit type and currency, stop-loss shape) and generates the rest.
 */
export const vmBuildTemplate = (goal, shell, templateOptions, symbols = []) => {
  const template = vmNormalizeTemplate(goal, templateOptions);
  if (!(template && isRecord(shell))) {
    return clone(shell);
  }
  const specGoal = goal === 'scalper' ? 'scalper' : 'long';
  const specs = PROFILE_SPECS[specGoal];
  const spec = specs[template.profile] ?? Object.values(specs)[0];
  const out = clone(shell);
  const quote = quoteOf(shell);
  const settings = isRecord(shell.settings) ? clone(shell.settings) : { type: 'CUSTOM', includePosition: true };
  const indentType = typeof settings.indentType === 'string' ? settings.indentType : null;
  const grid = spec.grid.indents ? explicitGrid(spec.grid.indents, spec.grid.volumes, indentType) : vmGrid(spec.grid, indentType);
  settings.baseOrder = grid.baseOrder;
  settings.orders = grid.orders;
  delete settings.conditionGroups;
  out.settings = settings;

  const filters = templateFilters(goal, template.profile, template.timeframe);
  out.conditions = filters.entry.map((condition) => ({ ...condition }));
  delete out.conditionGroups;

  const shellProfit = isRecord(shell.profit) ? shell.profit : null;
  const profitType = shellProfit && typeof shellProfit.type === 'string' ? shellProfit.type : 'PERCENT';
  out.profit = {
    ...(shellProfit ?? {}),
    type: profitType,
    currency: (shellProfit && shellProfit.currency) || quote,
    checkPnl: spec.takeProfit,
    conditions: profitType === 'SIGNAL' ? filters.exit.map((condition) => ({ ...condition })) : [],
  };
  delete out.profit.conditionGroups;

  if (template.stopLoss && Number.isFinite(spec.stopLoss)) {
    const shellStop = isRecord(shell.stopLoss) ? shell.stopLoss : null;
    out.stopLoss = {
      ...(shellStop ?? {}),
      indent: spec.stopLoss,
      termination: shellStop && typeof shellStop.termination === 'boolean' ? shellStop.termination : false,
      conditionalIndent: null,
      conditionalIndentType: shellStop ? (shellStop.conditionalIndentType ?? null) : null,
      conditions: shellStop && Array.isArray(shellStop.conditions) ? [] : null,
    };
    delete out.stopLoss.conditionGroups;
  } else {
    out.stopLoss = null;
  }
  out.pullUp = Number.isFinite(spec.pullUp) ? spec.pullUp : null;
  out.algorithm = 'LONG';
  const coins = templateSymbols(symbols, quote);
  if (coins.length > 0) {
    out.symbols = coins;
  }
  return template.direction === 'SHORT' ? vmMirrorToShort(out).strategy : out;
};

/**
 * Base strategy of a run: the bot itself (improve, coins), the mirrored bot (short "mirror"), or a
 * template built on it. `notes` describes what was generated (for the log).
 */
export const vmPrepareBase = (config, shell) => {
  const goal = config?.goal ?? 'improve';
  if (!vmIsTemplateGoal(goal) || !isRecord(shell)) {
    return { strategy: shell, notes: null };
  }
  const template = vmNormalizeTemplate(goal, config.template);
  const symbols = Array.isArray(config.symbols) ? config.symbols : [];
  if (goal === 'short' && template.profile === 'mirror') {
    const { strategy, notes } = vmMirrorToShort(shell);
    const quote = quoteOf(shell);
    const coins = templateSymbols(symbols, quote);
    if (coins.length > 0) {
      strategy.symbols = coins;
    }
    return { strategy, notes: { kind: 'mirror', ...notes } };
  }
  return { strategy: vmBuildTemplate(goal, shell, template, symbols), notes: { kind: 'template', profile: template.profile } };
};

// ═══════════════════════════════════════════════════════════════════
// SEARCH SPACE
// ═══════════════════════════════════════════════════════════════════

/** Indicators the template modes may swap filters to (all with explicit thresholds). */
const VALUED_POOL = Object.freeze([
  'RSI',
  'CCI',
  'MFI',
  'WILLIAMS_R',
  'CMO',
  'CONNORS_RSI',
  'PRICE_CHANGE',
  'PRICE_CHANGE_PERIOD_10',
  'ADX',
  'ATR_PERCENT',
  'BOLLINGER_BANDS_WIDTH',
  'VOLUME_SPIKE_DETECTOR_14',
]);

/** Structure pool of a mode (null = the catalogue's popular indicators, as in "improve"). */
export const vmStructurePool = (goal, direction = 'LONG') => {
  if (goal === 'scalper' || goal === 'short' || direction === 'SHORT') {
    return [...VALUED_POOL];
  }
  return null;
};

/** Search scope of a template mode (2.15 scope flags). */
export const vmScope = (goal, baseScope, direction = 'LONG') => ({
  ...baseScope,
  dcaIndents: true,
  dcaVolumes: true,
  takeProfit: true,
  stopLoss: true,
  stopLossConditional: false,
  pullUp: goal !== 'scalper',
  leverage: false,
  entryThresholds: true,
  orderThresholds: true,
  exitThresholds: true,
  timeframes: goal === 'scalper',
  linkOrderConditions: true,
  rangeWidthPercent: 100,
  structureSwap: true,
  structureRemove: true,
  structureAdd: 1,
  structurePlaces: { entry: true, order: false, takeProfit: false, stopLoss: false },
  structureSimilarOnly: false,
  structurePool: vmStructurePool(goal, direction),
});

const getAtPath = (object, path) => {
  let current = object;
  for (const key of path ?? []) {
    if (current === null || current === undefined) {
      return undefined;
    }
    current = current[key];
  }
  return current;
};

/** Absolute threshold windows of entry/exit filters per direction of the comparison. */
const THRESHOLD_WINDOWS = Object.freeze({
  RSI: { below: [8, 45], above: [55, 92] },
  MFI: { below: [5, 45], above: [55, 95] },
  CONNORS_RSI: { below: [3, 35], above: [65, 97] },
  STOCHASTIC: { below: [3, 40], above: [60, 97] },
  CCI: { below: [-350, -30], above: [30, 350] },
  CMO: { below: [-90, -10], above: [10, 90] },
  WILLIAMS_R: { below: [-99, -55], above: [-45, -1] },
});

const isBelow = (operation) => operation === 'LESS' || operation === 'LESS_OR_EQUAL';
const isAbove = (operation) => operation === 'GREATER' || operation === 'GREATER_OR_EQUAL';

/**
 * Range overrides (`config.ranges` format `{[dimId]: {enabled, min, max}}`) of a template mode for the
 * draft dimensions `drafts` (2.15 `che` output) of the template strategy `strategy`. Template modes
 * search globally, so take-profit, stop-loss and filter thresholds get absolute windows instead of
 * ranges around the template value.
 */
export const vmModeRanges = (goal, drafts, strategy) => {
  if (!vmIsTemplateGoal(goal) || !Array.isArray(drafts)) {
    return {};
  }
  const ranges = {};
  const set = (dim, min, max) => {
    const low = Math.max(dim.hardMin ?? Number.NEGATIVE_INFINITY, Math.min(min, dim.baseValue));
    const high = Math.min(dim.hardMax ?? Number.POSITIVE_INFINITY, Math.max(max, dim.baseValue));
    if (high > low) {
      ranges[dim.id] = { enabled: true, min: roundTo(low, 6), max: roundTo(high, 6) };
    }
  };
  for (const dim of drafts) {
    if (!isRecord(dim) || typeof dim.id !== 'string') {
      continue;
    }
    switch (dim.kind) {
      case 'takeProfit':
        if (goal === 'scalper') {
          set(dim, 0.2, 1.5);
        } else {
          set(dim, 0.3, 3);
        }
        break;
      case 'stopLoss':
        if (goal === 'scalper') {
          set(dim, 0.5, 6);
        } else {
          set(dim, 3, 35);
        }
        break;
      case 'pullUp':
        if (goal !== 'scalper') {
          set(dim, 0.1, 1.5);
        }
        break;
      case 'orderIndent':
        if ((dim.orderIndex ?? 0) >= 1 && Number.isFinite(dim.baseValue) && dim.baseValue > 0) {
          set(dim, dim.baseValue * 0.4, dim.baseValue * 2.2);
        }
        break;
      case 'orderVolume':
        // The entry order must stay meaningful: a scalper enters with most of its volume, a DCA bot
        // with a small share (otherwise the search drifts to "enter with 1 % and average").
        if ((dim.orderIndex ?? 0) === 0) {
          if (goal === 'scalper') {
            set(dim, 25, 80);
          } else {
            set(dim, 2, 30);
          }
        }
        break;
      case 'conditionValue': {
        const target = Array.isArray(dim.targets) ? dim.targets[0] : null;
        const condition = target ? getAtPath(strategy, target.path) : null;
        const window = isRecord(condition) ? THRESHOLD_WINDOWS[vmIndicatorBase(condition.indicator)] : null;
        if (window) {
          if (isBelow(condition.operation)) {
            set(dim, window.below[0], window.below[1]);
          } else if (isAbove(condition.operation)) {
            set(dim, window.above[0], window.above[1]);
          }
        }
        break;
      }
      default:
        break;
    }
  }
  return ranges;
};

// ═══════════════════════════════════════════════════════════════════
// RUN RECIPES
// ═══════════════════════════════════════════════════════════════════

/** Default period preset of a mode: scalpers on minute timeframes are checked on the last year. */
export const vmPeriodPreset = (goal) => (goal === 'scalper' ? '1y' : 'since2020');

/** Form fields a fresh tab of a mode starts with (merged under the saved form). */
export const vmFormDefaults = (goal) =>
  goal === 'improve' || !VM_GOAL_IDS.includes(goal)
    ? { goal: 'improve' }
    : { goal, template: vmDefaultTemplate(goal), periodPreset: vmPeriodPreset(goal) };

/**
 * Form of a mode's one-click run (the template modes' counterpart of 2.15 `Fft`). `presets` are the
 * engine's depth presets (`z$.portfolio`), `periodOf(preset)` returns `{from, to}`.
 */
export const vmSimpleForm = (goal, form, { presets, periodOf, symbols = [] }) => {
  const template = vmNormalizeTemplate(goal, form.template);
  const periodPreset =
    form.periodPreset && form.periodPreset !== 'custom' ? form.periodPreset : vmPeriodPreset(goal);
  const period = periodOf(periodPreset);
  const balanced = presets?.balanced ?? {};
  const base = {
    ...form,
    goal,
    mode: 'portfolio',
    symbolsText: symbols.length > 0 && !form.symbolsText.trim() ? symbols.join(', ') : form.symbolsText,
    periodPreset,
    periodFrom: period.from,
    periodTo: period.to,
    portfolioTarget: VM_STABLE_METRIC,
    constraints: {
      ...form.constraints,
      minDealsSharePercent: 0,
      maxPortfolioDrawdownPercent: form.constraints?.maxPortfolioDrawdownPercent ?? 35,
      vtMaxAvgHoldHours: goal === 'scalper' ? (form.constraints?.vtMaxAvgHoldHours ?? 24) : null,
      vtMinDealsPerCoinMonth: goal === 'scalper' ? (form.constraints?.vtMinDealsPerCoinMonth ?? 2) : null,
    },
    portfolio: {
      ...form.portfolio,
      screeningAuto: true,
      screeningCount: 12,
      screeningMonths: 0,
      finalists: 3,
      maxDepositChangePercent: 0,
      foldConsistency: true,
    },
  };
  if (goal === 'coins') {
    return {
      ...base,
      template: null,
      genetic: { ...form.genetic, ...balanced, searchMethod: 'pattern' },
      robustness: { ...form.robustness, enabled: false },
    };
  }
  return {
    ...base,
    template,
    genetic: { ...form.genetic, ...balanced, searchMethod: 'pattern', searchBudget: 140 },
    scope: vmScope(goal, form.scope, template?.direction ?? 'LONG'),
    robustness: { ...form.robustness, enabled: true, neighbors: Math.max(4, form.robustness?.neighbors ?? 4) },
  };
};

// ═══════════════════════════════════════════════════════════════════
// DEAL STATISTICS, CONSTRAINTS, SCORE
// ═══════════════════════════════════════════════════════════════════

/**
 * Deal style of a portfolio (from the engine's compact deals `{s, e, n, m, st, o}`): average and
 * median holding time of closed deals, deals per coin and month, share of deals that used the grid.
 */
export const vmDealStats = (entries) => {
  let closed = 0;
  let total = 0;
  let withGrid = 0;
  let gridOrders = 0;
  let holdSum = 0;
  const holds = [];
  let coinMonths = 0;
  for (const entry of entries ?? []) {
    const from = Date.parse(entry.from);
    const to = Date.parse(entry.to);
    if (Number.isFinite(from) && Number.isFinite(to) && to > from) {
      coinMonths += (to - from) / DAY_MS / MONTH_DAYS;
    }
    for (const deal of entry.deals ?? []) {
      total += 1;
      const fills = Array.isArray(deal.o) ? deal.o.filter((value) => value > 0).length : 0;
      if (fills > 0) {
        withGrid += 1;
        gridOrders += fills;
      }
      if (deal.st === 'F' && Number.isFinite(deal.s) && Number.isFinite(deal.e) && deal.e >= deal.s) {
        const hours = (deal.e - deal.s) / HOUR_MS;
        closed += 1;
        holdSum += hours;
        holds.push(hours);
      }
    }
  }
  return {
    deals: total,
    closedDeals: closed,
    avgHoldHours: closed > 0 ? roundTo(holdSum / closed, 2) : 0,
    medianHoldHours: closed > 0 ? roundTo(vmMedian(holds), 2) : 0,
    dealsPerCoinMonth: coinMonths > EPS ? roundTo(total / coinMonths, 3) : 0,
    gridShare: total > 0 ? roundTo(withGrid / total, 4) : 0,
    avgGridOrders: withGrid > 0 ? roundTo(gridOrders / withGrid, 2) : 0,
  };
};

/** Mode constraints on the deal style → `[{message, amount}]` (amount adds to the violation size). */
export const vmStyleViolations = (stats, constraints) => {
  const out = [];
  if (!stats || !constraints) {
    return out;
  }
  const maxHold = constraints.vtMaxAvgHoldHours;
  if (Number.isFinite(maxHold) && maxHold > 0 && stats.closedDeals > 0 && stats.avgHoldHours > maxHold) {
    out.push({
      message: `Средняя сделка ${stats.avgHoldHours.toFixed(1)} ч — дольше ${maxHold} ч`,
      amount: (stats.avgHoldHours - maxHold) / maxHold,
    });
  }
  const minRate = constraints.vtMinDealsPerCoinMonth;
  if (Number.isFinite(minRate) && minRate > 0 && stats.dealsPerCoinMonth < minRate) {
    out.push({
      message: `Сделок ${stats.dealsPerCoinMonth.toFixed(2)} в месяц на монету — меньше ${minRate}`,
      amount: (minRate - stats.dealsPerCoinMonth) / minRate,
    });
  }
  return out;
};

/** Metric ids handled by this module. */
export const vmIsModeMetric = (metric) => metric === VM_STABLE_METRIC;

const RATIO_SOFT_CAP = 50;
const RATIO_MAX = 100;

/**
 * Profit-to-drawdown with a floor for the drawdown. Up to 50 it is the plain ratio; above that it
 * is compressed smoothly towards 100 (never a flat cap), so very high ratios — a tiny drawdown
 * on a long period — still order the variants instead of all tying at the limit.
 */
export const vmProfitToRisk = (profit, risk, floor = 1e-6) => {
  if (!(risk > floor)) {
    return profit > 0 ? RATIO_MAX : 0;
  }
  const value = profit / risk;
  const magnitude = Math.abs(value);
  if (magnitude <= RATIO_SOFT_CAP) {
    return value;
  }
  const span = RATIO_MAX - RATIO_SOFT_CAP;
  const soft = RATIO_SOFT_CAP + span * (1 - Math.exp(-(magnitude - RATIO_SOFT_CAP) / span));
  return Math.sign(value) * soft;
};

/**
 * Stable profit-to-drawdown: the portfolio's profit / drawdown, discounted by the share of folds
 * (half-year slices and coin groups of the search) that are profitable: × (0.5 + 0.5 × share).
 * A strategy that earns only in one period or on one group of coins scores up to 2× lower.
 */
export const vmStableScore = (raw, folds) => {
  const overall = vmProfitToRisk(raw?.adjustedProfit ?? 0, raw?.risk ?? 0);
  if (!(overall > 0) || !Array.isArray(folds) || folds.length < 2) {
    return roundTo(overall, 4);
  }
  const positive = folds.filter((fold) => fold.profit > 0).length;
  return roundTo(overall * (0.5 + (0.5 * positive) / folds.length), 4);
};

/** Score of a portfolio fitness for a mode metric. */
export const vmScore = (metric, { raw, folds }) => (metric === VM_STABLE_METRIC ? vmStableScore(raw, folds) : 0);

// ═══════════════════════════════════════════════════════════════════
// ROBUSTNESS VERDICT (absolute)
// ═══════════════════════════════════════════════════════════════════

const ABSOLUTE_THRESHOLDS = Object.freeze({
  robustYearsShare: 0.8,
  partialYearsShare: 0.5,
  robustRetention: 0.6,
  partialRetention: 0.4,
});

/**
 * Verdict of a template-mode finalist from the 2.15 robustness report (segments carry the
 * candidate's `profit`): instead of "not worse than the original" it asks "profitable on its own"
 * — in most years, on coins outside the search, before the search period, on the stress list — and
 * whether nearby settings keep most of the score (plateau median retention).
 */
export const vmAbsoluteVerdict = (report) => {
  if (!isRecord(report) || !Array.isArray(report.segments)) {
    return report;
  }
  const years = report.segments.filter((segment) => segment.kind === 'year');
  const profitableYears = years.filter((segment) => segment.profit > 0);
  const yearsShare = years.length > 0 ? profitableYears.length / years.length : 1;
  const others = report.segments.filter(
    (segment) => segment.kind === 'coins' || segment.kind === 'time' || segment.kind === 'stress',
  );
  const othersProfitable = others.filter((segment) => segment.profit > 0);
  const retention = report.plateau && report.plateau.tested > 0 ? report.plateau.medianRetention : null;
  const reasons = [];
  const lossYears = years.filter((segment) => !(segment.profit > 0));
  if (lossYears.length > 0) {
    reasons.push(`убыток в ${lossYears.length} из ${years.length} лет: ${lossYears.map((segment) => segment.label).join(', ')}`);
  }
  for (const segment of others) {
    if (!(segment.profit > 0)) {
      reasons.push(`${segment.label}: убыток`);
    }
  }
  if (retention !== null && retention < ABSOLUTE_THRESHOLDS.robustRetention) {
    reasons.push(`при небольшом сдвиге настроек оценка падает до ${Math.round(retention * 100)}% — острый пик`);
  }
  const { robustYearsShare, partialYearsShare, robustRetention, partialRetention } = ABSOLUTE_THRESHOLDS;
  let verdict = 'overfit';
  if (
    yearsShare >= robustYearsShare - EPS &&
    othersProfitable.length === others.length &&
    (retention === null || retention >= robustRetention - EPS)
  ) {
    verdict = 'robust';
  } else if (
    yearsShare >= partialYearsShare - EPS &&
    othersProfitable.length * 2 >= others.length &&
    (retention === null || retention >= partialRetention - EPS)
  ) {
    verdict = 'partial';
  }
  return { ...report, verdict, reasons, yearsNotWorse: profitableYears.length, absolute: true };
};

// ═══════════════════════════════════════════════════════════════════
// COIN SELECTION
// ═══════════════════════════════════════════════════════════════════

/**
 * Best subset of coins for a fixed strategy, chosen offline from its deals (no extra backtests).
 * `evaluate(entries)` is the portfolio aggregation (blocking by bots, dynamic limit) returning
 * `{profit, risk, monthlyPercent, drawdownPercent, deals}`. Greedy backward elimination on the
 * stable profit-to-drawdown, then a forward pass that re-adds coins that help.
 * `folds(entries)` (optional) returns fold metrics `[{profit}]` for the stability factor.
 * Each elimination step tries only the `candidatesPerStep` coins with the lowest contribution
 * (measured once on the full set), so the cost stays linear in the number of coins.
 */
export const vmSelectCoins = (
  entries,
  evaluate,
  { minCoins = 5, maxRemoveShare = 0.6, minGain = 0.005, folds = null, candidatesPerStep = 12 } = {},
) => {
  const usable = (entries ?? []).filter((entry) => Array.isArray(entry.deals) && entry.deals.length > 0);
  const score = (subset) => {
    if (subset.length === 0) {
      return { score: Number.NEGATIVE_INFINITY, metrics: null };
    }
    const metrics = evaluate(subset);
    const raw = { adjustedProfit: metrics.profit, risk: metrics.risk };
    return { score: vmStableScore(raw, folds ? folds(subset) : null), metrics };
  };
  const all = score(usable);
  const perCoin = usable.map((entry) => {
    const without = usable.filter((item) => item !== entry);
    const result = score(without);
    const net = entry.deals.reduce((sum, deal) => sum + (deal.st === 'F' ? deal.n : 0), 0);
    return {
      symbol: entry.symbol,
      deals: entry.deals.length,
      net: roundTo(net, 4),
      maxMae: roundTo(entry.deals.reduce((max, deal) => Math.max(max, deal.m ?? 0), 0), 4),
      contribution: Number.isFinite(result.score) ? roundTo(all.score - result.score, 4) : 0,
    };
  });
  const floor = Math.max(1, Math.min(usable.length, Math.max(minCoins, Math.ceil(usable.length * (1 - maxRemoveShare)))));
  let current = [...usable];
  let currentScore = all.score;
  const removed = [];
  const contribution = new Map(perCoin.map((item) => [item.symbol, item.contribution]));
  while (current.length > floor) {
    let best = null;
    const candidates = [...current]
      .sort((a, b) => (contribution.get(a.symbol) ?? 0) - (contribution.get(b.symbol) ?? 0))
      .slice(0, Math.max(1, candidatesPerStep));
    for (const entry of candidates) {
      const candidate = score(current.filter((item) => item !== entry));
      if (!best || candidate.score > best.score) {
        best = { entry, score: candidate.score };
      }
    }
    if (!best || !(best.score > currentScore + Math.max(minGain * Math.abs(currentScore), EPS))) {
      break;
    }
    removed.push({ symbol: best.entry.symbol, gain: roundTo(best.score - currentScore, 4) });
    current = current.filter((item) => item !== best.entry);
    currentScore = best.score;
  }
  // Re-add coins that improve the subset (the greedy order may have removed a useful one).
  for (const entry of usable) {
    if (current.includes(entry)) {
      continue;
    }
    const candidate = score([...current, entry]);
    if (candidate.score > currentScore + Math.max(minGain * Math.abs(currentScore), EPS)) {
      current = [...current, entry];
      currentScore = candidate.score;
      const index = removed.findIndex((item) => item.symbol === entry.symbol);
      if (index >= 0) {
        removed.splice(index, 1);
      }
    }
  }
  const selected = score(current);
  const order = new Map(usable.map((entry, index) => [entry.symbol, index]));
  return {
    selected: current.map((entry) => entry.symbol).sort((a, b) => order.get(a) - order.get(b)),
    removed,
    noDeals: (entries ?? []).filter((entry) => !usable.includes(entry)).map((entry) => entry.symbol),
    all: all.metrics ? { ...all.metrics, score: roundTo(all.score, 4) } : null,
    best: selected.metrics ? { ...selected.metrics, score: roundTo(selected.score, 4) } : null,
    perCoin: perCoin.sort((a, b) => a.contribution - b.contribution),
  };
};
