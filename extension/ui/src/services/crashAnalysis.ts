/**
 * Сервис анализа ликвидационных проливов BTC
 * 
 * Функционал:
 * 1. Объединение данных из разных источников
 * 2. Feature engineering (производные признаки)
 * 3. Расчёт целевой переменной (crash_next_6h)
 * 4. Статистический анализ (корреляции, логистическая регрессия)
 * 5. Поиск правил и walk-forward валидация
 */

import {
  fetchFearGreedIndex,
  fetchFundingRates,
  fetchKlines,
  fetchOpenInterest,
  fetchSpotKlines,
  fetchTopTraderLongShortRatio,
  formatTimestamp,
  getCurrentTimestamp,
  getStartTimestamp,
} from '../api/binanceData';
import type { FearGreedData } from '../api/binanceData';
import type {
  CombinedRule,
  CrashAnalysisConfig,
  CrashAnalysisResult,
  DataBar,
  DataLoadProgress,
  FeatureBar,
  FeatureCorrelation,
  FeatureImportance,
  FundingRateBar,
  OpenInterestBar,
  RawBar,
  ThresholdRule,
  WalkForwardResult,
} from '../types/crashAnalysis';

// ═══════════════════════════════════════════════════════════════════
// ТИПЫ ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ
// ═══════════════════════════════════════════════════════════════════

interface FearGreedBar {
  timestamp: number;
  value: number;
  classification: string;
}

interface BasisBar {
  timestamp: number;
  basisPercent: number;
}

// ═══════════════════════════════════════════════════════════════════
// УТИЛИТЫ СТАТИСТИКИ
// ═══════════════════════════════════════════════════════════════════

/**
 * Расчёт среднего
 */
const mean = (arr: number[]): number => {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
};

/**
 * Расчёт стандартного отклонения
 */
const std = (arr: number[]): number => {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
};

/**
 * Расчёт Z-score для значения относительно массива
 */
const zscore = (value: number, arr: number[]): number => {
  const m = mean(arr);
  const s = std(arr);
  if (s === 0) return 0;
  return (value - m) / s;
};

/**
 * Расчёт корреляции Пирсона
 */
const pearsonCorrelation = (x: number[], y: number[]): number => {
  if (x.length !== y.length || x.length < 2) return 0;
  
  const n = x.length;
  const meanX = mean(x);
  const meanY = mean(y);
  
  let numerator = 0;
  let denomX = 0;
  let denomY = 0;
  
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    numerator += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }
  
  const denom = Math.sqrt(denomX * denomY);
  if (denom === 0) return 0;
  
  return numerator / denom;
};

/**
 * Приблизительный расчёт p-value для корреляции
 * Использует t-распределение
 */
const correlationPValue = (r: number, n: number): number => {
  if (n < 3) return 1;
  const t = r * Math.sqrt((n - 2) / (1 - r * r));
  // Приближённый p-value (двусторонний тест)
  const df = n - 2;
  // Используем нормальное приближение для больших df
  if (df > 30) {
    const z = Math.abs(t);
    return 2 * (1 - normalCDF(z));
  }
  // Грубое приближение для малых df
  return 2 * (1 - tCDF(Math.abs(t), df));
};

/**
 * CDF стандартного нормального распределения (приближение)
 */
const normalCDF = (x: number): number => {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  
  return 0.5 * (1.0 + sign * y);
};

/**
 * CDF t-распределения (грубое приближение)
 */
const tCDF = (t: number, df: number): number => {
  // Используем нормальное приближение для простоты
  const z = t * Math.sqrt(df / (df - 2));
  return normalCDF(z);
};

// ═══════════════════════════════════════════════════════════════════
// РАСЧЁТ ATR
// ═══════════════════════════════════════════════════════════════════

/**
 * Расчёт True Range
 */
const trueRange = (high: number, low: number, prevClose: number): number => {
  return Math.max(
    high - low,
    Math.abs(high - prevClose),
    Math.abs(low - prevClose),
  );
};

/**
 * Расчёт ATR для массива баров
 */
const calculateATR = (bars: RawBar[], period: number): number[] => {
  const atr: number[] = new Array(bars.length).fill(null);
  
  if (bars.length < period + 1) return atr;
  
  // Первый ATR = простое среднее TR
  let sumTR = 0;
  for (let i = 1; i <= period; i++) {
    sumTR += trueRange(bars[i].high, bars[i].low, bars[i - 1].close);
  }
  atr[period] = sumTR / period;
  
  // Последующие ATR = EMA-style
  for (let i = period + 1; i < bars.length; i++) {
    const tr = trueRange(bars[i].high, bars[i].low, bars[i - 1].close);
    atr[i] = (atr[i - 1]! * (period - 1) + tr) / period;
  }
  
  return atr;
};

// ═══════════════════════════════════════════════════════════════════
// ОБЪЕДИНЕНИЕ ДАННЫХ
// ═══════════════════════════════════════════════════════════════════

/**
 * Интерполяция funding rate на 15-минутные интервалы
 * Funding происходит каждые 8 часов, нужно заполнить промежутки
 */
const interpolateFundingRates = (
  fundingRates: FundingRateBar[],
  timestamps: number[],
): Map<number, number> => {
  const result = new Map<number, number>();
  
  if (fundingRates.length === 0) return result;
  
  // Сортируем по времени
  const sorted = [...fundingRates].sort((a, b) => a.timestamp - b.timestamp);
  
  let fundingIdx = 0;
  let currentFunding = sorted[0].fundingRate;
  
  for (const ts of timestamps) {
    // Находим ближайший funding rate
    while (fundingIdx < sorted.length - 1 && sorted[fundingIdx + 1].timestamp <= ts) {
      fundingIdx++;
      currentFunding = sorted[fundingIdx].fundingRate;
    }
    result.set(ts, currentFunding);
  }
  
  return result;
};

/**
 * Интерполяция Open Interest на 15-минутные интервалы
 */
const interpolateOpenInterest = (
  oi: OpenInterestBar[],
  timestamps: number[],
): Map<number, number> => {
  const result = new Map<number, number>();
  
  if (oi.length === 0) return result;
  
  const sorted = [...oi].sort((a, b) => a.timestamp - b.timestamp);
  
  let oiIdx = 0;
  let currentOI = sorted[0].openInterest;
  
  for (const ts of timestamps) {
    while (oiIdx < sorted.length - 1 && sorted[oiIdx + 1].timestamp <= ts) {
      oiIdx++;
      currentOI = sorted[oiIdx].openInterest;
    }
    result.set(ts, currentOI);
  }
  
  return result;
};

/**
 * Интерполяция Fear & Greed Index на 15-минутные интервалы
 * Fear & Greed публикуется 1 раз в сутки, интерполируем на все бары дня
 */
const interpolateFearGreedIndex = (
  fearGreedData: FearGreedBar[],
  timestamps: number[],
): Map<number, number> => {
  const result = new Map<number, number>();
  
  if (fearGreedData.length === 0) return result;
  
  const sorted = [...fearGreedData].sort((a, b) => a.timestamp - b.timestamp);
  
  let fgIdx = 0;
  let currentFG = sorted[0].value;
  
  for (const ts of timestamps) {
    // Найти соответствующий дневной Fear & Greed
    while (fgIdx < sorted.length - 1 && sorted[fgIdx + 1].timestamp <= ts) {
      fgIdx++;
      currentFG = sorted[fgIdx].value;
    }
    result.set(ts, currentFG);
  }
  
  return result;
};

/**
 * Расчёт Spot-Futures Basis для 15-минутных интервалов
 * Basis = (Futures Price - Spot Price) / Spot Price * 100
 */
const calculateSpotFuturesBasis = (
  futuresKlines: RawBar[],
  spotKlines: RawBar[],
): Map<number, number> => {
  const result = new Map<number, number>();
  
  if (spotKlines.length === 0) return result;
  
  // Создаём map спот цен по timestamp
  const spotPriceMap = new Map<number, number>();
  for (const bar of spotKlines) {
    spotPriceMap.set(bar.timestamp, bar.close);
  }
  
  // Для каждого фьючерсного бара ищем ближайший спот
  for (const futuresBar of futuresKlines) {
    // Ищем точное совпадение или ближайший спот бар
    let spotPrice = spotPriceMap.get(futuresBar.timestamp);
    
    if (spotPrice === undefined) {
      // Если нет точного совпадения, ищем ближайший предыдущий
      let closestTs = 0;
      for (const ts of spotPriceMap.keys()) {
        if (ts <= futuresBar.timestamp && ts > closestTs) {
          closestTs = ts;
        }
      }
      if (closestTs > 0) {
        spotPrice = spotPriceMap.get(closestTs);
      }
    }
    
    if (spotPrice !== undefined && spotPrice > 0) {
      const basis = ((futuresBar.close - spotPrice) / spotPrice) * 100;
      result.set(futuresBar.timestamp, basis);
    }
  }
  
  return result;
};

/**
 * Объединение всех данных в DataBar[]
 */
const mergeData = (
  klines: RawBar[],
  fundingRates: FundingRateBar[],
  openInterest: OpenInterestBar[],
  fearGreedData: FearGreedBar[] = [],
  spotKlines: RawBar[] = [],
): DataBar[] => {
  const timestamps = klines.map((k) => k.timestamp);
  const fundingMap = interpolateFundingRates(fundingRates, timestamps);
  const oiMap = interpolateOpenInterest(openInterest, timestamps);
  const fearGreedMap = interpolateFearGreedIndex(fearGreedData, timestamps);
  const basisMap = calculateSpotFuturesBasis(klines, spotKlines);
  const atr14 = calculateATR(klines, 14);
  
  return klines.map((k, i) => ({
    timestamp: k.timestamp,
    priceClose: k.close,
    priceVolume: k.quoteVolume,
    openInterest: oiMap.get(k.timestamp) ?? null,
    fundingRate: fundingMap.get(k.timestamp) ?? null,
    takerBuyVolume: k.takerBuyVolume,
    takerSellVolume: k.takerSellVolume,
    atr14: atr14[i] ?? null,
    fearGreedIndex: fearGreedMap.get(k.timestamp) ?? null,
    spotFuturesBasis: basisMap.get(k.timestamp) ?? null,
    // Placeholders для недоступных данных
    exchangeBtcInflow: null,
    exchangeBtcReserve: null,
    bidDepthTop20Usd: null,
    askDepthTop20Usd: null,
    liquidationClusterBelowPriceUsd: null,
  }));
};

// ═══════════════════════════════════════════════════════════════════
// FEATURE ENGINEERING
// ═══════════════════════════════════════════════════════════════════

/**
 * Расчёт производных признаков
 */
const calculateFeatures = (
  data: DataBar[],
  config: CrashAnalysisConfig,
): FeatureBar[] => {
  const features: FeatureBar[] = [];
  
  for (let i = 0; i < data.length; i++) {
    const bar = data[i];
    
    // Получаем исторические окна для z-score
    const window24h = data.slice(Math.max(0, i - config.zscore24hBars), i);
    const window72h = data.slice(Math.max(0, i - config.zscore72hBars), i);
    const window1h = data.slice(Math.max(0, i - 4), i); // 4 бара = 1 час при 15m
    const window4h = data.slice(Math.max(0, i - 16), i);
    const window7d = data.slice(Math.max(0, i - 672), i); // 672 бара = 7 дней при 15m
    
    // OI Z-score 24h
    let oiZscore24h: number | null = null;
    if (bar.openInterest !== null && window24h.length >= 10) {
      const oiValues = window24h.filter((b) => b.openInterest !== null).map((b) => b.openInterest!);
      if (oiValues.length >= 10) {
        oiZscore24h = zscore(bar.openInterest, oiValues);
      }
    }
    
    // Funding Z-score 24h
    let fundingZscore24h: number | null = null;
    if (bar.fundingRate !== null && window24h.length >= 10) {
      const fundingValues = window24h.filter((b) => b.fundingRate !== null).map((b) => b.fundingRate!);
      if (fundingValues.length >= 10) {
        fundingZscore24h = zscore(bar.fundingRate, fundingValues);
      }
    }
    
    // ATR Z-score 72h
    let atrZscore72h: number | null = null;
    if (bar.atr14 !== null && window72h.length >= 50) {
      const atrValues = window72h.filter((b) => b.atr14 !== null).map((b) => b.atr14!);
      if (atrValues.length >= 50) {
        atrZscore72h = zscore(bar.atr14, atrValues);
      }
    }
    
    // Fear & Greed Z-score 7d
    let fearGreedZscore7d: number | null = null;
    let fearGreedExtreme: -1 | 0 | 1 | null = null;
    if (bar.fearGreedIndex !== null && window7d.length >= 100) {
      const fgValues = window7d.filter((b) => b.fearGreedIndex !== null).map((b) => b.fearGreedIndex!);
      if (fgValues.length >= 100) {
        fearGreedZscore7d = zscore(bar.fearGreedIndex, fgValues);
      }
      // Extreme Fear < 20, Extreme Greed > 80 (как бинарный признак)
      fearGreedExtreme = bar.fearGreedIndex <= 20 ? -1 : bar.fearGreedIndex >= 80 ? 1 : 0;
    }
    
    // Basis Z-score 24h
    let basisZscore24h: number | null = null;
    let basisNegative: number | null = null;
    if (bar.spotFuturesBasis !== null && window24h.length >= 10) {
      const basisValues = window24h.filter((b) => b.spotFuturesBasis !== null).map((b) => b.spotFuturesBasis!);
      if (basisValues.length >= 10) {
        basisZscore24h = zscore(bar.spotFuturesBasis, basisValues);
      }
      // Negative basis (backwardation) — медвежий сигнал
      basisNegative = bar.spotFuturesBasis < 0 ? 1 : 0;
    }
    
    // Taker delta ratio
    const takerDeltaRatio = bar.takerBuyVolume > 0 
      ? bar.takerSellVolume / bar.takerBuyVolume 
      : 1;
    
    // Price changes
    const priceChangePct1h = window1h.length > 0 
      ? ((bar.priceClose - window1h[0].priceClose) / window1h[0].priceClose) * 100 
      : 0;
    const priceChangePct4h = window4h.length > 0 
      ? ((bar.priceClose - window4h[0].priceClose) / window4h[0].priceClose) * 100 
      : 0;
    const priceChangePct24h = window24h.length > 0 
      ? ((bar.priceClose - window24h[0].priceClose) / window24h[0].priceClose) * 100 
      : 0;
    
    // Volume Z-score 24h
    let volumeZscore24h = 0;
    if (window24h.length >= 10) {
      const volumeValues = window24h.map((b) => b.priceVolume);
      volumeZscore24h = zscore(bar.priceVolume, volumeValues);
    }
    
    features.push({
      ...bar,
      oiZscore24h,
      fundingZscore24h,
      atrZscore72h,
      fearGreedZscore7d,
      fearGreedExtreme,
      basisZscore24h,
      basisNegative,
      bidDepthDeltaPct1h: null, // Недоступно
      takerDeltaRatio,
      exchangeInflowZscore24h: null, // Недоступно
      reserveDelta24h: null, // Недоступно
      liquidationDensityRatio: null, // Недоступно
      priceChangePct1h,
      priceChangePct4h,
      priceChangePct24h,
      volumeZscore24h,
      crashNext6h: null, // Будет рассчитано отдельно
    });
  }
  
  return features;
};

/**
 * Расчёт целевой переменной crash_next_6h
 */
const calculateTargetVariable = (
  features: FeatureBar[],
  config: CrashAnalysisConfig,
): FeatureBar[] => {
  const result = [...features];
  
  for (let i = 0; i < result.length; i++) {
    // Для последних баров где нет будущих данных - null
    if (i + config.crashWindowBars >= result.length) {
      result[i].crashNext6h = null;
      continue;
    }
    
    const currentPrice = result[i].priceClose;
    const threshold = currentPrice * (1 - config.crashThresholdPct / 100);
    
    // Проверяем был ли crash в следующих N барах
    let crash = false;
    for (let j = 1; j <= config.crashWindowBars; j++) {
      if (result[i + j].priceClose <= threshold) {
        crash = true;
        break;
      }
    }
    
    result[i].crashNext6h = crash ? 1 : 0;
  }
  
  return result;
};

// ═══════════════════════════════════════════════════════════════════
// СТАТИСТИЧЕСКИЙ АНАЛИЗ
// ═══════════════════════════════════════════════════════════════════

/**
 * Расчёт корреляций признаков с целевой переменной
 */
const calculateCorrelations = (features: FeatureBar[]): FeatureCorrelation[] => {
  // Фильтруем бары с известным crash
  const validBars = features.filter((f) => f.crashNext6h !== null);
  const target = validBars.map((f) => f.crashNext6h!);
  
  const featureNames: Array<{ name: string; getter: (f: FeatureBar) => number | null }> = [
    { name: 'oiZscore24h', getter: (f) => f.oiZscore24h },
    { name: 'fundingZscore24h', getter: (f) => f.fundingZscore24h },
    { name: 'atrZscore72h', getter: (f) => f.atrZscore72h },
    { name: 'fearGreedZscore7d', getter: (f) => f.fearGreedZscore7d },
    { name: 'fearGreedExtreme', getter: (f) => f.fearGreedExtreme },
    { name: 'basisZscore24h', getter: (f) => f.basisZscore24h },
    { name: 'basisNegative', getter: (f) => f.basisNegative },
    { name: 'takerDeltaRatio', getter: (f) => f.takerDeltaRatio },
    { name: 'priceChangePct1h', getter: (f) => f.priceChangePct1h },
    { name: 'priceChangePct4h', getter: (f) => f.priceChangePct4h },
    { name: 'priceChangePct24h', getter: (f) => f.priceChangePct24h },
    { name: 'volumeZscore24h', getter: (f) => f.volumeZscore24h },
  ];
  
  const correlations: FeatureCorrelation[] = [];
  
  for (const { name, getter } of featureNames) {
    const values: number[] = [];
    const targets: number[] = [];
    
    for (let i = 0; i < validBars.length; i++) {
      const value = getter(validBars[i]);
      if (value !== null) {
        values.push(value);
        targets.push(target[i]);
      }
    }
    
    if (values.length < 30) continue;
    
    const corr = pearsonCorrelation(values, targets);
    const pValue = correlationPValue(corr, values.length);
    
    correlations.push({
      featureName: name,
      correlation: corr,
      pValue,
      isSignificant: pValue < 0.05,
    });
  }
  
  return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
};

/**
 * Простой расчёт feature importance через univariate selection
 */
const calculateFeatureImportance = (features: FeatureBar[]): FeatureImportance[] => {
  const correlations = calculateCorrelations(features);
  
  return correlations.map((c, i) => ({
    featureName: c.featureName,
    importance: Math.abs(c.correlation),
    rank: i + 1,
  }));
};

/**
 * Поиск оптимальных пороговых правил
 */
const findThresholdRules = (
  features: FeatureBar[],
  config: CrashAnalysisConfig,
): CombinedRule[] => {
  const validBars = features.filter((f) => f.crashNext6h !== null);
  const baseCrashRate = mean(validBars.map((f) => f.crashNext6h!));
  
  const featureGetters: Array<{ name: string; getter: (f: FeatureBar) => number | null }> = [
    { name: 'oiZscore24h', getter: (f) => f.oiZscore24h },
    { name: 'fundingZscore24h', getter: (f) => f.fundingZscore24h },
    { name: 'atrZscore72h', getter: (f) => f.atrZscore72h },
    { name: 'fearGreedZscore7d', getter: (f) => f.fearGreedZscore7d },
    { name: 'fearGreedExtreme', getter: (f) => f.fearGreedExtreme },
    { name: 'basisZscore24h', getter: (f) => f.basisZscore24h },
    { name: 'basisNegative', getter: (f) => f.basisNegative },
    { name: 'takerDeltaRatio', getter: (f) => f.takerDeltaRatio },
    { name: 'priceChangePct4h', getter: (f) => f.priceChangePct4h },
    { name: 'volumeZscore24h', getter: (f) => f.volumeZscore24h },
  ];
  
  const rules: CombinedRule[] = [];
  
  // Одиночные правила
  for (const { name, getter } of featureGetters) {
    const values = validBars
      .map((f, i) => ({ value: getter(f), crash: f.crashNext6h!, idx: i }))
      .filter((v) => v.value !== null) as Array<{ value: number; crash: number; idx: number }>;
    
    if (values.length < 100) continue;
    
    // Пробуем разные пороги (квантили)
    const sortedValues = [...values].sort((a, b) => a.value - b.value);
    const quantiles = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    
    for (const q of quantiles) {
      const threshold = sortedValues[Math.floor(sortedValues.length * q)].value;
      
      // Правило: feature > threshold
      const matchingAbove = values.filter((v) => v.value > threshold);
      if (matchingAbove.length >= config.minRuleSupport) {
        const crashes = matchingAbove.filter((v) => v.crash === 1).length;
        const prob = crashes / matchingAbove.length;
        const lift = prob / baseCrashRate;
        
        if (lift > 1.5) {
          rules.push({
            conditions: [{ featureName: name, operator: '>', threshold }],
            crashProbability: prob,
            support: matchingAbove.length,
            crashes,
            lift,
          });
        }
      }
      
      // Правило: feature < threshold
      const matchingBelow = values.filter((v) => v.value < threshold);
      if (matchingBelow.length >= config.minRuleSupport) {
        const crashes = matchingBelow.filter((v) => v.crash === 1).length;
        const prob = crashes / matchingBelow.length;
        const lift = prob / baseCrashRate;
        
        if (lift > 1.5) {
          rules.push({
            conditions: [{ featureName: name, operator: '<', threshold }],
            crashProbability: prob,
            support: matchingBelow.length,
            crashes,
            lift,
          });
        }
      }
    }
  }
  
  // Сортируем по lift
  rules.sort((a, b) => b.lift - a.lift);
  
  // Комбинируем лучшие правила
  const combinedRules: CombinedRule[] = [];
  const topSingleRules = rules.slice(0, 10);
  
  // Пробуем комбинации из 2-3 правил
  for (let i = 0; i < topSingleRules.length; i++) {
    for (let j = i + 1; j < topSingleRules.length; j++) {
      const rule1 = topSingleRules[i];
      const rule2 = topSingleRules[j];
      
      // Проверяем что признаки разные
      if (rule1.conditions[0].featureName === rule2.conditions[0].featureName) continue;
      
      // Находим бары где выполняются оба условия
      const matching = validBars.filter((f) => {
        const check1 = checkCondition(f, rule1.conditions[0]);
        const check2 = checkCondition(f, rule2.conditions[0]);
        return check1 && check2;
      });
      
      if (matching.length >= config.minRuleSupport) {
        const crashes = matching.filter((f) => f.crashNext6h === 1).length;
        const prob = crashes / matching.length;
        const lift = prob / baseCrashRate;
        
        if (lift > 2.0) {
          combinedRules.push({
            conditions: [...rule1.conditions, ...rule2.conditions],
            crashProbability: prob,
            support: matching.length,
            crashes,
            lift,
          });
        }
      }
    }
  }
  
  // Объединяем и сортируем
  const allRules = [...rules, ...combinedRules];
  allRules.sort((a, b) => b.lift - a.lift);
  
  return allRules.slice(0, 20);
};

/**
 * Проверка условия для бара
 */
const checkCondition = (bar: FeatureBar, condition: ThresholdRule): boolean => {
  const getterMap: Record<string, (f: FeatureBar) => number | null> = {
    oiZscore24h: (f) => f.oiZscore24h,
    fundingZscore24h: (f) => f.fundingZscore24h,
    atrZscore72h: (f) => f.atrZscore72h,
    fearGreedZscore7d: (f) => f.fearGreedZscore7d,
    fearGreedExtreme: (f) => f.fearGreedExtreme,
    basisZscore24h: (f) => f.basisZscore24h,
    basisNegative: (f) => f.basisNegative,
    takerDeltaRatio: (f) => f.takerDeltaRatio,
    priceChangePct1h: (f) => f.priceChangePct1h,
    priceChangePct4h: (f) => f.priceChangePct4h,
    priceChangePct24h: (f) => f.priceChangePct24h,
    volumeZscore24h: (f) => f.volumeZscore24h,
  };
  
  const getter = getterMap[condition.featureName];
  if (!getter) return false;
  
  const value = getter(bar);
  if (value === null) return false;
  
  switch (condition.operator) {
    case '>': return value > condition.threshold;
    case '<': return value < condition.threshold;
    case '>=': return value >= condition.threshold;
    case '<=': return value <= condition.threshold;
    default: return false;
  }
};

/**
 * Walk-forward валидация
 */
const walkForwardValidation = (
  features: FeatureBar[],
  config: CrashAnalysisConfig,
): WalkForwardResult[] => {
  const results: WalkForwardResult[] = [];
  
  // Определяем периоды
  const sortedBars = [...features].sort((a, b) => a.timestamp - b.timestamp);
  const firstYear = new Date(sortedBars[0].timestamp).getFullYear();
  const lastYear = new Date(sortedBars[sortedBars.length - 1].timestamp).getFullYear();
  
  // Делаем валидацию по годам
  for (let testYear = firstYear + config.trainYears; testYear <= lastYear; testYear++) {
    const trainStart = new Date(testYear - config.trainYears, 0, 1).getTime();
    const trainEnd = new Date(testYear, 0, 1).getTime();
    const testStart = trainEnd;
    const testEnd = new Date(testYear + config.testYears, 0, 1).getTime();
    
    const trainBars = features.filter(
      (f) => f.timestamp >= trainStart && f.timestamp < trainEnd && f.crashNext6h !== null,
    );
    const testBars = features.filter(
      (f) => f.timestamp >= testStart && f.timestamp < testEnd && f.crashNext6h !== null,
    );
    
    if (trainBars.length < 1000 || testBars.length < 100) continue;
    
    // Находим правила на train
    const trainRules = findThresholdRules(trainBars, config);
    
    // Тестируем на test
    const trainCrashRate = mean(trainBars.map((f) => f.crashNext6h!));
    const testCrashRate = mean(testBars.map((f) => f.crashNext6h!));
    
    // Оцениваем лучшее правило на test
    let testAccuracy = 0;
    let testPrecision = 0;
    let testRecall = 0;
    
    if (trainRules.length > 0) {
      const bestRule = trainRules[0];
      
      let tp = 0, fp = 0, tn = 0, fn = 0;
      
      for (const bar of testBars) {
        const predicted = bestRule.conditions.every((c) => checkCondition(bar, c)) ? 1 : 0;
        const actual = bar.crashNext6h!;
        
        if (predicted === 1 && actual === 1) tp++;
        else if (predicted === 1 && actual === 0) fp++;
        else if (predicted === 0 && actual === 0) tn++;
        else fn++;
      }
      
      testAccuracy = (tp + tn) / testBars.length;
      testPrecision = tp + fp > 0 ? tp / (tp + fp) : 0;
      testRecall = tp + fn > 0 ? tp / (tp + fn) : 0;
    }
    
    results.push({
      trainPeriod: { from: formatTimestamp(trainStart), to: formatTimestamp(trainEnd) },
      testPeriod: { from: formatTimestamp(testStart), to: formatTimestamp(testEnd) },
      trainCrashRate,
      testCrashRate,
      rules: trainRules.slice(0, 5),
      testAccuracy,
      testPrecision,
      testRecall,
    });
  }
  
  return results;
};

// ═══════════════════════════════════════════════════════════════════
// ГЛАВНЫЙ АНАЛИЗ
// ═══════════════════════════════════════════════════════════════════

export interface AnalysisCallbacks {
  onProgress: (progress: DataLoadProgress) => void;
}

/**
 * Запуск полного анализа
 */
export const runCrashAnalysis = async (
  config: CrashAnalysisConfig,
  callbacks: AnalysisCallbacks,
): Promise<CrashAnalysisResult> => {
  const { onProgress } = callbacks;
  
  const symbol = 'BTCUSDT';
  const timeframe = config.timeframe || '1h';
  const periodYears = config.periodYears || 2;
  const startTime = getStartTimestamp(periodYears);
  const endTime = getCurrentTimestamp();
  
  // Вспомогательная функция для предотвращения блокировки UI
  const yieldToUI = () => new Promise<void>(resolve => setTimeout(resolve, 0));
  
  // 1. Загрузка OHLCV
  onProgress({ stage: 'loading-ohlcv', progress: 0, message: `Загрузка OHLCV (${timeframe}, ${periodYears} лет)...` });
  await yieldToUI();
  
  const klines = await fetchKlines(symbol, timeframe, startTime, endTime, async (loaded, pct) => {
    onProgress({ stage: 'loading-ohlcv', progress: pct, message: `Загружено ${loaded} баров...` });
    if (loaded % 5000 === 0) await yieldToUI(); // Yield каждые 5000 баров
  });
  
  await yieldToUI();
  
  // 2. Загрузка Funding Rate
  onProgress({ stage: 'loading-funding', progress: 0, message: 'Загрузка Funding Rate...' });
  await yieldToUI();
  
  const fundingRates = await fetchFundingRates(symbol, startTime, endTime, async (loaded) => {
    onProgress({ stage: 'loading-funding', progress: 50, message: `Загружено ${loaded} записей funding...` });
  });
  
  await yieldToUI();
  
  // 3. Загрузка Open Interest (ограниченная история)
  onProgress({ stage: 'loading-oi', progress: 0, message: 'Загрузка Open Interest...' });
  
  const openInterest = await fetchOpenInterest(symbol, timeframe, startTime, endTime, async (loaded) => {
    onProgress({ stage: 'loading-oi', progress: 50, message: `Загружено ${loaded} записей OI...` });
  });
  
  await yieldToUI();
  
  // 4. Загрузка Fear & Greed Index
  onProgress({ stage: 'loading-feargreed', progress: 0, message: 'Загрузка Fear & Greed Index...' });
  
  let fearGreedData: FearGreedBar[] = [];
  try {
    const rawFearGreed = await fetchFearGreedIndex();
    fearGreedData = rawFearGreed.map((fg) => ({
      timestamp: fg.timestamp, // API уже возвращает миллисекунды
      value: fg.value,
      classification: fg.classification,
    }));
    onProgress({ stage: 'loading-feargreed', progress: 100, message: `Загружено ${fearGreedData.length} записей Fear & Greed` });
  } catch (error) {
    console.warn('Failed to fetch Fear & Greed Index:', error);
    onProgress({ stage: 'loading-feargreed', progress: 100, message: 'Fear & Greed недоступен, продолжаем без него' });
  }
  
  // 5. Загрузка Spot BTCUSDT для расчёта Basis
  onProgress({ stage: 'loading-spot', progress: 0, message: 'Загрузка Spot BTCUSDT для расчёта Basis...' });
  await yieldToUI();
  
  let spotKlines: typeof klines = [];
  try {
    spotKlines = await fetchSpotKlines(symbol, timeframe, startTime, endTime, async (loaded, pct) => {
      onProgress({ stage: 'loading-spot', progress: pct, message: `Загружено ${loaded} спот баров...` });
      if (loaded % 5000 === 0) await yieldToUI();
    });
  } catch (error) {
    console.warn('Failed to fetch Spot klines:', error);
    onProgress({ stage: 'loading-spot', progress: 100, message: 'Спот данные недоступны, продолжаем без Basis' });
  }
  
  await yieldToUI();
  
  // 6-10. Обработка данных в Web Worker (не блокирует UI)
  onProgress({ stage: 'processing', progress: 0, message: 'Запуск обработки в фоновом потоке...' });
  
  const workerResult = await runInWorker({
    klines,
    fundingRates,
    openInterest,
    fearGreed: fearGreedData,
    spotKlines,
    config: {
      crashThresholdPct: config.crashThresholdPct,
      crashWindowBars: config.crashWindowBars,
      zscore24hBars: config.zscore24hBars,
      zscore72hBars: config.zscore72hBars,
      minRuleSupport: config.minRuleSupport,
      maxRuleConditions: config.maxRuleConditions,
      trainYears: config.trainYears,
      testYears: config.testYears,
    },
  }, (stage, progress, message) => {
    onProgress({ stage: stage as DataLoadProgress['stage'], progress, message });
  });
  
  // 10. Формирование результата
  onProgress({ stage: 'done', progress: 100, message: 'Анализ завершён!' });
  
  // Формируем рекомендации
  const recommendations: string[] = [];
  
  if (workerResult.topRules.length > 0) {
    const bestRule = workerResult.topRules[0];
    recommendations.push(
      `Лучшее правило: ${bestRule.conditions.map((c) => `${c.featureName} ${c.operator} ${c.threshold.toFixed(2)}`).join(' И ')}`,
    );
    recommendations.push(
      `Вероятность падения ≥${config.crashThresholdPct}% при срабатывании: ${(bestRule.crashProbability * 100).toFixed(1)}%`,
    );
    recommendations.push(
      `Lift относительно базовой частоты: ${bestRule.lift.toFixed(2)}x`,
    );
  }
  
  if (workerResult.correlations.length > 0) {
    const topCorr = workerResult.correlations[0];
    recommendations.push(
      `Наиболее коррелирующий признак: ${topCorr.featureName} (r=${topCorr.correlation.toFixed(3)})`,
    );
  }
  
  return {
    datasetInfo: {
      symbol,
      timeframe,
      periodFrom: formatTimestamp(startTime),
      periodTo: formatTimestamp(endTime),
      totalBars: workerResult.datasetInfo.totalBars,
      barsWithCrash: workerResult.datasetInfo.barsWithCrash,
      baseCrashRate: workerResult.datasetInfo.baseCrashRate,
    },
    correlations: workerResult.correlations,
    featureImportance: workerResult.featureImportance,
    topRules: workerResult.topRules,
    walkForwardResults: workerResult.walkForwardResults,
    recommendations,
    features: workerResult.features,
  };
};

/**
 * Запуск обработки в Web Worker
 */
const runInWorker = (
  payload: WorkerPayload,
  onProgress: (stage: string, progress: number, message: string) => void
): Promise<WorkerResultData> => {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL('../workers/crashAnalysisWorker.ts', import.meta.url),
      { type: 'module' }
    );
    
    worker.onmessage = (event) => {
      const { type, ...data } = event.data;
      
      if (type === 'progress') {
        onProgress(data.stage, data.progress, data.message);
      } else if (type === 'result') {
        worker.terminate();
        resolve(data.data);
      } else if (type === 'error') {
        worker.terminate();
        reject(new Error(data.error));
      }
    };
    
    worker.onerror = (error) => {
      worker.terminate();
      reject(error);
    };
    
    worker.postMessage({ type: 'process', payload });
  });
};

interface WorkerPayload {
  klines: RawBar[];
  fundingRates: FundingRateBar[];
  openInterest: OpenInterestBar[];
  fearGreed: FearGreedBar[];
  spotKlines: RawBar[];
  config: {
    crashThresholdPct: number;
    crashWindowBars: number;
    zscore24hBars: number;
    zscore72hBars: number;
    minRuleSupport: number;
    maxRuleConditions: number;
    trainYears: number;
    testYears: number;
  };
}

interface WorkerResultData {
  correlations: FeatureCorrelation[];
  featureImportance: FeatureImportance[];
  topRules: CombinedRule[];
  walkForwardResults: WalkForwardResult[];
  features: FeatureBar[];
  datasetInfo: {
    totalBars: number;
    barsWithCrash: number;
    baseCrashRate: number;
  };
}

/**
 * Экспорт датасета в CSV
 */
export const exportDatasetToCSV = (features: FeatureBar[]): string => {
  const headers = [
    'timestamp',
    'priceClose',
    'priceVolume',
    'openInterest',
    'fundingRate',
    'takerBuyVolume',
    'takerSellVolume',
    'atr14',
    'fearGreedIndex',
    'spotFuturesBasis',
    'oiZscore24h',
    'fundingZscore24h',
    'atrZscore72h',
    'fearGreedZscore7d',
    'fearGreedExtreme',
    'basisZscore24h',
    'basisNegative',
    'takerDeltaRatio',
    'priceChangePct1h',
    'priceChangePct4h',
    'priceChangePct24h',
    'volumeZscore24h',
    'crashNext6h',
  ];
  
  const rows = features.map((f) => [
    new Date(f.timestamp).toISOString(),
    f.priceClose,
    f.priceVolume,
    f.openInterest ?? '',
    f.fundingRate ?? '',
    f.takerBuyVolume,
    f.takerSellVolume,
    f.atr14 ?? '',
    f.fearGreedIndex ?? '',
    f.spotFuturesBasis ?? '',
    f.oiZscore24h ?? '',
    f.fundingZscore24h ?? '',
    f.atrZscore72h ?? '',
    f.fearGreedZscore7d ?? '',
    f.fearGreedExtreme ?? '',
    f.basisZscore24h ?? '',
    f.basisNegative ?? '',
    f.takerDeltaRatio,
    f.priceChangePct1h,
    f.priceChangePct4h,
    f.priceChangePct24h,
    f.volumeZscore24h,
    f.crashNext6h ?? '',
  ].join(','));
  
  return [headers.join(','), ...rows].join('\n');
};
