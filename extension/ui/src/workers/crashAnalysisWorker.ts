/**
 * Web Worker для тяжёлых вычислений анализа crash'ей
 * 
 * Выносим в отдельный поток:
 * - Объединение данных (mergeData)
 * - Feature engineering (calculateFeatures)
 * - Расчёт целевой переменной
 * - Корреляции и feature importance
 * - Walk-forward валидация
 */

// Типы сообщений
interface WorkerMessage {
  type: 'process';
  payload: {
    klines: KlineData[];
    fundingRates: FundingData[];
    openInterest: OIData[];
    fearGreed: FearGreedData[];
    spotKlines: KlineData[];
    config: AnalysisConfig;
  };
}

interface KlineData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quoteVolume: number;
  takerBuyVolume: number;
  takerSellVolume: number;
}

interface FundingData {
  timestamp: number;
  fundingRate: number;
}

interface OIData {
  timestamp: number;
  openInterest: number;
  openInterestValue: number;
}

interface FearGreedData {
  timestamp: number;
  value: number;
  classification: string;
}

interface AnalysisConfig {
  crashThresholdPct: number;
  crashWindowBars: number;
  zscore24hBars: number;
  zscore72hBars: number;
  minRuleSupport: number;
  maxRuleConditions: number;
  trainYears: number;
  testYears: number;
}

interface DataBar {
  timestamp: number;
  priceClose: number;
  priceVolume: number;
  openInterest: number | null;
  fundingRate: number | null;
  takerBuyVolume: number;
  takerSellVolume: number;
  atr14: number | null;
  fearGreedIndex: number | null;
  spotFuturesBasis: number | null;
}

interface FeatureBar extends DataBar {
  oiZscore24h: number | null;
  fundingZscore24h: number | null;
  atrZscore72h: number | null;
  fearGreedZscore7d: number | null;
  fearGreedExtreme: -1 | 0 | 1 | null;
  basisZscore24h: number | null;
  basisNegative: number | null;
  takerDeltaRatio: number;
  priceChangePct1h: number;
  priceChangePct4h: number;
  priceChangePct24h: number;
  volumeZscore24h: number;
  crashNext6h: 0 | 1 | null;
}

// ═══════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

const mean = (arr: number[]): number => {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
};

const std = (arr: number[]): number => {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
};

const zscore = (value: number, arr: number[]): number => {
  const m = mean(arr);
  const s = std(arr);
  if (s === 0) return 0;
  return (value - m) / s;
};

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
  if (denomX === 0 || denomY === 0) return 0;
  return numerator / Math.sqrt(denomX * denomY);
};

// ═══════════════════════════════════════════════════════════════════
// MERGE DATA
// ═══════════════════════════════════════════════════════════════════

const mergeData = (
  klines: KlineData[],
  fundingRates: FundingData[],
  openInterest: OIData[],
  fearGreed: FearGreedData[],
  spotKlines: KlineData[]
): DataBar[] => {
  // Create lookup maps
  const fundingMap = new Map<number, number>();
  for (const fr of fundingRates) {
    fundingMap.set(fr.timestamp, fr.fundingRate);
  }
  
  const oiMap = new Map<number, number>();
  for (const oi of openInterest) {
    oiMap.set(oi.timestamp, oi.openInterest);
  }
  
  // Fear & Greed is daily - create day lookup
  const fgMap = new Map<string, number>();
  for (const fg of fearGreed) {
    const day = new Date(fg.timestamp).toISOString().split('T')[0];
    fgMap.set(day, fg.value);
  }
  
  // Spot prices for basis calculation
  const spotMap = new Map<number, number>();
  for (const spot of spotKlines) {
    spotMap.set(spot.timestamp, spot.close);
  }
  
  const result: DataBar[] = [];
  
  // Calculate ATR
  const atrPeriod = 14;
  const trueRanges: number[] = [];
  
  for (let i = 0; i < klines.length; i++) {
    const k = klines[i];
    const day = new Date(k.timestamp).toISOString().split('T')[0];
    
    // True Range
    let tr = k.high - k.low;
    if (i > 0) {
      const prevClose = klines[i - 1].close;
      tr = Math.max(
        k.high - k.low,
        Math.abs(k.high - prevClose),
        Math.abs(k.low - prevClose)
      );
    }
    trueRanges.push(tr);
    
    // ATR14
    let atr14: number | null = null;
    if (trueRanges.length >= atrPeriod) {
      atr14 = mean(trueRanges.slice(-atrPeriod));
    }
    
    // Spot-Futures Basis
    const spotPrice = spotMap.get(k.timestamp);
    let basis: number | null = null;
    if (spotPrice && spotPrice > 0) {
      basis = ((k.close - spotPrice) / spotPrice) * 100;
    }
    
    result.push({
      timestamp: k.timestamp,
      priceClose: k.close,
      priceVolume: k.quoteVolume,
      openInterest: oiMap.get(k.timestamp) ?? null,
      fundingRate: fundingMap.get(k.timestamp) ?? null,
      takerBuyVolume: k.takerBuyVolume,
      takerSellVolume: k.takerSellVolume,
      atr14,
      fearGreedIndex: fgMap.get(day) ?? null,
      spotFuturesBasis: basis,
    });
  }
  
  return result;
};

// ═══════════════════════════════════════════════════════════════════
// CALCULATE FEATURES (CHUNKED)
// ═══════════════════════════════════════════════════════════════════

const calculateFeatures = (
  data: DataBar[],
  config: AnalysisConfig,
  onProgress: (pct: number) => void
): FeatureBar[] => {
  const result: FeatureBar[] = [];
  const { zscore24hBars, zscore72hBars } = config;
  const barsIn1h = 4; // 15m timeframe
  const barsIn4h = 16;
  const barsIn7d = 96 * 7;
  
  const chunkSize = 5000;
  
  for (let i = 0; i < data.length; i++) {
    if (i % chunkSize === 0) {
      onProgress(Math.round((i / data.length) * 100));
    }
    
    const bar = data[i];
    
    // Z-scores
    let oiZscore24h: number | null = null;
    let fundingZscore24h: number | null = null;
    let atrZscore72h: number | null = null;
    let fearGreedZscore7d: number | null = null;
    let fearGreedExtreme: -1 | 0 | 1 | null = null;
    let basisZscore24h: number | null = null;
    let basisNegative: number | null = null;
    
    // OI Z-score
    if (i >= zscore24hBars && bar.openInterest !== null) {
      const oiWindow = data.slice(i - zscore24hBars, i)
        .map(d => d.openInterest)
        .filter((v): v is number => v !== null);
      if (oiWindow.length >= zscore24hBars / 2) {
        oiZscore24h = zscore(bar.openInterest, oiWindow);
      }
    }
    
    // Funding Z-score
    if (i >= zscore24hBars && bar.fundingRate !== null) {
      const frWindow = data.slice(i - zscore24hBars, i)
        .map(d => d.fundingRate)
        .filter((v): v is number => v !== null);
      if (frWindow.length >= 10) {
        fundingZscore24h = zscore(bar.fundingRate, frWindow);
      }
    }
    
    // ATR Z-score
    if (i >= zscore72hBars && bar.atr14 !== null) {
      const atrWindow = data.slice(i - zscore72hBars, i)
        .map(d => d.atr14)
        .filter((v): v is number => v !== null);
      if (atrWindow.length >= zscore72hBars / 2) {
        atrZscore72h = zscore(bar.atr14, atrWindow);
      }
    }
    
    // Fear & Greed Z-score
    if (i >= barsIn7d && bar.fearGreedIndex !== null) {
      const fgWindow = data.slice(i - barsIn7d, i)
        .map(d => d.fearGreedIndex)
        .filter((v): v is number => v !== null);
      if (fgWindow.length >= 50) {
        fearGreedZscore7d = zscore(bar.fearGreedIndex, fgWindow);
      }
      // Extreme fear/greed
      if (bar.fearGreedIndex <= 20) {
        fearGreedExtreme = -1;
      } else if (bar.fearGreedIndex >= 80) {
        fearGreedExtreme = 1;
      } else {
        fearGreedExtreme = 0;
      }
    }
    
    // Basis Z-score
    if (i >= zscore24hBars && bar.spotFuturesBasis !== null) {
      const basisWindow = data.slice(i - zscore24hBars, i)
        .map(d => d.spotFuturesBasis)
        .filter((v): v is number => v !== null);
      if (basisWindow.length >= zscore24hBars / 2) {
        basisZscore24h = zscore(bar.spotFuturesBasis, basisWindow);
      }
      basisNegative = bar.spotFuturesBasis < 0 ? 1 : 0;
    }
    
    // Taker delta ratio
    const takerDeltaRatio = bar.takerBuyVolume > 0 
      ? bar.takerSellVolume / bar.takerBuyVolume 
      : 1;
    
    // Price changes
    let priceChangePct1h = 0;
    let priceChangePct4h = 0;
    let priceChangePct24h = 0;
    
    if (i >= barsIn1h) {
      const prev = data[i - barsIn1h].priceClose;
      priceChangePct1h = ((bar.priceClose - prev) / prev) * 100;
    }
    if (i >= barsIn4h) {
      const prev = data[i - barsIn4h].priceClose;
      priceChangePct4h = ((bar.priceClose - prev) / prev) * 100;
    }
    if (i >= zscore24hBars) {
      const prev = data[i - zscore24hBars].priceClose;
      priceChangePct24h = ((bar.priceClose - prev) / prev) * 100;
    }
    
    // Volume Z-score
    let volumeZscore24h = 0;
    if (i >= zscore24hBars) {
      const volWindow = data.slice(i - zscore24hBars, i).map(d => d.priceVolume);
      volumeZscore24h = zscore(bar.priceVolume, volWindow);
    }
    
    result.push({
      ...bar,
      oiZscore24h,
      fundingZscore24h,
      atrZscore72h,
      fearGreedZscore7d,
      fearGreedExtreme,
      basisZscore24h,
      basisNegative,
      takerDeltaRatio,
      priceChangePct1h,
      priceChangePct4h,
      priceChangePct24h,
      volumeZscore24h,
      crashNext6h: null,
    });
  }
  
  return result;
};

// ═══════════════════════════════════════════════════════════════════
// TARGET VARIABLE
// ═══════════════════════════════════════════════════════════════════

const calculateTargetVariable = (
  features: FeatureBar[],
  config: AnalysisConfig
): FeatureBar[] => {
  const { crashThresholdPct, crashWindowBars } = config;
  
  for (let i = 0; i < features.length - crashWindowBars; i++) {
    const currentPrice = features[i].priceClose;
    let minPrice = currentPrice;
    
    for (let j = 1; j <= crashWindowBars; j++) {
      minPrice = Math.min(minPrice, features[i + j].priceClose);
    }
    
    const drawdown = ((currentPrice - minPrice) / currentPrice) * 100;
    features[i].crashNext6h = drawdown >= crashThresholdPct ? 1 : 0;
  }
  
  return features;
};

// ═══════════════════════════════════════════════════════════════════
// CORRELATIONS
// ═══════════════════════════════════════════════════════════════════

interface FeatureCorrelation {
  featureName: string;
  correlation: number;
  pValue: number;
  isSignificant: boolean;
}

const calculateCorrelations = (features: FeatureBar[]): FeatureCorrelation[] => {
  const validFeatures = features.filter(f => f.crashNext6h !== null);
  const target = validFeatures.map(f => f.crashNext6h as number);
  
  const featureNames: (keyof FeatureBar)[] = [
    'oiZscore24h', 'fundingZscore24h', 'atrZscore72h',
    'fearGreedZscore7d', 'basisZscore24h',
    'takerDeltaRatio', 'volumeZscore24h',
    'priceChangePct1h', 'priceChangePct4h', 'priceChangePct24h',
  ];
  
  const results: FeatureCorrelation[] = [];
  
  for (const name of featureNames) {
    const values = validFeatures
      .map(f => f[name])
      .filter((v): v is number => v !== null && !Number.isNaN(v));
    
    if (values.length < target.length * 0.5) continue;
    
    const validTarget = validFeatures
      .filter(f => f[name] !== null && !Number.isNaN(f[name] as number))
      .map(f => f.crashNext6h as number);
    
    const corr = pearsonCorrelation(values, validTarget);
    const n = values.length;
    const tStat = corr * Math.sqrt(n - 2) / Math.sqrt(1 - corr * corr);
    const pValue = 2 * (1 - normalCDF(Math.abs(tStat)));
    
    results.push({
      featureName: name,
      correlation: corr,
      pValue,
      isSignificant: pValue < 0.05,
    });
  }
  
  return results.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
};

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

// ═══════════════════════════════════════════════════════════════════
// FEATURE IMPORTANCE
// ═══════════════════════════════════════════════════════════════════

interface FeatureImportance {
  featureName: string;
  importance: number;
  rank: number;
}

const calculateFeatureImportance = (features: FeatureBar[]): FeatureImportance[] => {
  const correlations = calculateCorrelations(features);
  const total = correlations.reduce((sum, c) => sum + Math.abs(c.correlation), 0) || 1;
  
  return correlations.map((c, i) => ({
    featureName: c.featureName,
    importance: Math.abs(c.correlation) / total,
    rank: i + 1,
  }));
};

// ═══════════════════════════════════════════════════════════════════
// THRESHOLD RULES
// ═══════════════════════════════════════════════════════════════════

interface ThresholdRule {
  featureName: string;
  operator: '>' | '<' | '>=' | '<=';
  threshold: number;
}

interface CombinedRule {
  conditions: ThresholdRule[];
  crashProbability: number;
  support: number;
  crashes: number;
  lift: number;
}

const findThresholdRules = (
  features: FeatureBar[],
  config: AnalysisConfig
): CombinedRule[] => {
  const validFeatures = features.filter(f => f.crashNext6h !== null);
  const baseCrashRate = validFeatures.filter(f => f.crashNext6h === 1).length / validFeatures.length;
  
  const featureNames: (keyof FeatureBar)[] = [
    'oiZscore24h', 'fundingZscore24h', 'atrZscore72h',
    'takerDeltaRatio', 'volumeZscore24h',
    'priceChangePct1h', 'priceChangePct4h', 'priceChangePct24h',
  ];
  
  const rules: CombinedRule[] = [];
  
  // Single feature rules
  for (const name of featureNames) {
    const values = validFeatures
      .map(f => f[name])
      .filter((v): v is number => v !== null && !Number.isNaN(v));
    
    if (values.length < 100) continue;
    
    const sorted = [...values].sort((a, b) => a - b);
    const percentiles = [10, 25, 75, 90].map(p => sorted[Math.floor(sorted.length * p / 100)]);
    
    for (const threshold of percentiles) {
      for (const operator of ['>' as const, '<' as const]) {
        const matching = validFeatures.filter(f => {
          const val = f[name];
          if (val === null) return false;
          return operator === '>' ? val > threshold : val < threshold;
        });
        
        if (matching.length < config.minRuleSupport) continue;
        
        const crashes = matching.filter(f => f.crashNext6h === 1).length;
        const prob = crashes / matching.length;
        const lift = prob / baseCrashRate;
        
        if (lift > 1.2) {
          rules.push({
            conditions: [{ featureName: name, operator, threshold }],
            crashProbability: prob,
            support: matching.length,
            crashes,
            lift,
          });
        }
      }
    }
  }
  
  return rules
    .sort((a, b) => b.lift - a.lift)
    .slice(0, 20);
};

// ═══════════════════════════════════════════════════════════════════
// WALK-FORWARD VALIDATION
// ═══════════════════════════════════════════════════════════════════

interface WalkForwardResult {
  trainPeriod: { from: string; to: string };
  testPeriod: { from: string; to: string };
  trainCrashRate: number;
  testCrashRate: number;
  rules: CombinedRule[];
  testAccuracy: number;
  testPrecision: number;
  testRecall: number;
}

const walkForwardValidation = (
  features: FeatureBar[],
  config: AnalysisConfig
): WalkForwardResult[] => {
  const validFeatures = features.filter(f => f.crashNext6h !== null);
  const barsPerYear = 96 * 365; // 15m bars per year
  
  const trainBars = config.trainYears * barsPerYear;
  const testBars = config.testYears * barsPerYear;
  
  if (validFeatures.length < trainBars + testBars) {
    return [];
  }
  
  const results: WalkForwardResult[] = [];
  let startIdx = 0;
  
  while (startIdx + trainBars + testBars <= validFeatures.length) {
    const trainData = validFeatures.slice(startIdx, startIdx + trainBars);
    const testData = validFeatures.slice(startIdx + trainBars, startIdx + trainBars + testBars);
    
    const trainCrashRate = trainData.filter(f => f.crashNext6h === 1).length / trainData.length;
    const testCrashRate = testData.filter(f => f.crashNext6h === 1).length / testData.length;
    
    // Find rules on train data
    const rules = findThresholdRules(trainData as FeatureBar[], config);
    
    // Evaluate on test data
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (const bar of testData) {
      const bestRule = rules[0];
      let predicted = 0;
      
      if (bestRule) {
        const condition = bestRule.conditions[0];
        const val = bar[condition.featureName as keyof FeatureBar];
        if (typeof val === 'number') {
          if (condition.operator === '>') {
            predicted = val > condition.threshold ? 1 : 0;
          } else {
            predicted = val < condition.threshold ? 1 : 0;
          }
        }
      }
      
      const actual = bar.crashNext6h as number;
      if (predicted === 1 && actual === 1) tp++;
      else if (predicted === 1 && actual === 0) fp++;
      else if (predicted === 0 && actual === 0) tn++;
      else fn++;
    }
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    
    results.push({
      trainPeriod: {
        from: new Date(trainData[0].timestamp).toISOString().split('T')[0],
        to: new Date(trainData[trainData.length - 1].timestamp).toISOString().split('T')[0],
      },
      testPeriod: {
        from: new Date(testData[0].timestamp).toISOString().split('T')[0],
        to: new Date(testData[testData.length - 1].timestamp).toISOString().split('T')[0],
      },
      trainCrashRate,
      testCrashRate,
      rules,
      testAccuracy: accuracy,
      testPrecision: precision,
      testRecall: recall,
    });
    
    startIdx += testBars; // Move by test period
  }
  
  return results;
};

// ═══════════════════════════════════════════════════════════════════
// WORKER MESSAGE HANDLER
// ═══════════════════════════════════════════════════════════════════

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;
  
  if (type === 'process') {
    try {
      // Step 1: Merge data
      self.postMessage({ type: 'progress', stage: 'processing', progress: 0, message: 'Объединение данных...' });
      const dataBars = mergeData(
        payload.klines,
        payload.fundingRates,
        payload.openInterest,
        payload.fearGreed,
        payload.spotKlines
      );
      
      // Step 2: Feature engineering
      self.postMessage({ type: 'progress', stage: 'processing', progress: 20, message: 'Расчёт признаков...' });
      let features = calculateFeatures(dataBars, payload.config, (pct) => {
        self.postMessage({ 
          type: 'progress', 
          stage: 'processing', 
          progress: 20 + pct * 0.3, 
          message: `Расчёт признаков (${pct}%)...` 
        });
      });
      
      // Step 3: Target variable
      self.postMessage({ type: 'progress', stage: 'processing', progress: 50, message: 'Расчёт целевой переменной...' });
      features = calculateTargetVariable(features, payload.config);
      
      // Step 4: Correlations
      self.postMessage({ type: 'progress', stage: 'analyzing', progress: 60, message: 'Анализ корреляций...' });
      const correlations = calculateCorrelations(features);
      
      // Step 5: Feature importance
      self.postMessage({ type: 'progress', stage: 'analyzing', progress: 70, message: 'Расчёт важности признаков...' });
      const featureImportance = calculateFeatureImportance(features);
      
      // Step 6: Rules
      self.postMessage({ type: 'progress', stage: 'analyzing', progress: 80, message: 'Поиск правил...' });
      const topRules = findThresholdRules(features, payload.config);
      
      // Step 7: Walk-forward
      self.postMessage({ type: 'progress', stage: 'analyzing', progress: 90, message: 'Walk-forward валидация...' });
      const walkForwardResults = walkForwardValidation(features, payload.config);
      
      // Done
      const validBars = features.filter(f => f.crashNext6h !== null);
      const barsWithCrash = validBars.filter(f => f.crashNext6h === 1).length;
      
      self.postMessage({
        type: 'result',
        data: {
          correlations,
          featureImportance,
          topRules,
          walkForwardResults,
          features,
          datasetInfo: {
            totalBars: validBars.length,
            barsWithCrash,
            baseCrashRate: barsWithCrash / validBars.length,
          },
        },
      });
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
};

export {};
