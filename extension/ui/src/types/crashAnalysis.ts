/**
 * Типы данных для анализа ликвидационных проливов BTC
 */

// ═══════════════════════════════════════════════════════════════════
// БАЗОВЫЕ ДАННЫЕ
// ═══════════════════════════════════════════════════════════════════

/**
 * Сырой 15-минутный бар с данными от Binance
 */
export interface RawBar {
  timestamp: number; // Unix timestamp в ms
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number; // Объём в BTC
  quoteVolume: number; // Объём в USDT
  takerBuyVolume: number; // Объём покупок taker
  takerSellVolume: number; // Объём продаж taker
}

/**
 * Данные Open Interest
 */
export interface OpenInterestBar {
  timestamp: number;
  openInterest: number; // В BTC
  openInterestValue: number; // В USDT
}

/**
 * Данные Funding Rate
 */
export interface FundingRateBar {
  timestamp: number;
  fundingRate: number; // Обычно от -0.01 до 0.01
}

/**
 * Агрегированный бар с основными данными
 */
export interface DataBar {
  timestamp: number;
  
  // OHLCV
  priceClose: number;
  priceVolume: number; // Quote volume (USDT)
  
  // Open Interest
  openInterest: number | null;
  
  // Funding Rate
  fundingRate: number | null;
  
  // Taker volumes
  takerBuyVolume: number;
  takerSellVolume: number;
  
  // ATR (рассчитывается)
  atr14: number | null;
  
  // Fear & Greed Index (0-100)
  fearGreedIndex: number | null;
  
  // Spot-Futures Basis (%)
  spotFuturesBasis: number | null;
  
  // Placeholder для on-chain данных (если станут доступны)
  exchangeBtcInflow: number | null;
  exchangeBtcReserve: number | null;
  
  // Placeholder для глубины стакана
  bidDepthTop20Usd: number | null;
  askDepthTop20Usd: number | null;
  
  // Placeholder для ликвидаций
  liquidationClusterBelowPriceUsd: number | null;
}

// ═══════════════════════════════════════════════════════════════════
// ПРОИЗВОДНЫЕ ПРИЗНАКИ (FEATURES)
// ═══════════════════════════════════════════════════════════════════

/**
 * Производные признаки для анализа
 */
export interface FeatureBar extends DataBar {
  // Z-scores
  oiZscore24h: number | null;
  fundingZscore24h: number | null;
  atrZscore72h: number | null;
  
  // Fear & Greed
  fearGreedZscore7d: number | null;
  fearGreedExtreme: -1 | 0 | 1 | null; // -1=extreme fear, 0=neutral, 1=extreme greed
  
  // Spot-Futures Basis
  basisZscore24h: number | null;
  basisNegative: number | null; // 1 если basis < 0 (медвежий сигнал), 0 иначе
  
  // Deltas
  bidDepthDeltaPct1h: number | null;
  takerDeltaRatio: number; // taker_sell / taker_buy
  
  // On-chain (placeholders)
  exchangeInflowZscore24h: number | null;
  reserveDelta24h: number | null;
  
  // Ликвидации (placeholder)
  liquidationDensityRatio: number | null;
  
  // Дополнительные признаки
  priceChangePct1h: number;
  priceChangePct4h: number;
  priceChangePct24h: number;
  volumeZscore24h: number;
  
  // Целевая переменная
  crashNext6h: 0 | 1 | null; // null для последних баров где ещё неизвестно
}

// ═══════════════════════════════════════════════════════════════════
// РЕЗУЛЬТАТЫ АНАЛИЗА
// ═══════════════════════════════════════════════════════════════════

/**
 * Корреляция признака с целевой переменной
 */
export interface FeatureCorrelation {
  featureName: string;
  correlation: number; // Pearson
  pValue: number;
  isSignificant: boolean;
}

/**
 * Важность признака (feature importance)
 */
export interface FeatureImportance {
  featureName: string;
  importance: number;
  rank: number;
}

/**
 * Пороговое правило
 */
export interface ThresholdRule {
  featureName: string;
  operator: '>' | '<' | '>=' | '<=';
  threshold: number;
}

/**
 * Комбинированное правило (несколько условий)
 */
export interface CombinedRule {
  conditions: ThresholdRule[];
  crashProbability: number; // P(crash | conditions)
  support: number; // Сколько раз условия выполнялись
  crashes: number; // Сколько раз после условий был crash
  lift: number; // Во сколько раз вероятность выше базовой
}

/**
 * Результаты walk-forward валидации
 */
export interface WalkForwardResult {
  trainPeriod: { from: string; to: string };
  testPeriod: { from: string; to: string };
  trainCrashRate: number;
  testCrashRate: number;
  rules: CombinedRule[];
  testAccuracy: number;
  testPrecision: number;
  testRecall: number;
}

/**
 * Полный результат анализа
 */
export interface CrashAnalysisResult {
  // Метаданные
  datasetInfo: {
    symbol: string;
    timeframe: string;
    periodFrom: string;
    periodTo: string;
    totalBars: number;
    barsWithCrash: number;
    baseCrashRate: number;
  };
  
  // Корреляции
  correlations: FeatureCorrelation[];
  
  // Feature importance
  featureImportance: FeatureImportance[];
  
  // Лучшие правила
  topRules: CombinedRule[];
  
  // Walk-forward валидация
  walkForwardResults: WalkForwardResult[];
  
  // Итоговые рекомендации
  recommendations: string[];
  
  // Датасет с признаками (для ML)
  features: FeatureBar[];
}

// ═══════════════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ АНАЛИЗА
// ═══════════════════════════════════════════════════════════════════

export type AnalysisTimeframe = '15m' | '1h' | '4h';
export type AnalysisPeriod = 1 | 2 | 3 | 5;

export interface CrashAnalysisConfig {
  // Параметры загрузки данных
  timeframe: AnalysisTimeframe;
  periodYears: AnalysisPeriod;
  
  // Параметры целевой переменной
  crashThresholdPct: number; // Порог падения (по умолчанию 7%)
  crashWindowBars: number; // Окно поиска crash (по умолчанию 24 бара = 6 часов)
  
  // Параметры feature engineering
  zscore24hBars: number; // 96 баров = 24 часа при 15m таймфрейме
  zscore72hBars: number; // 288 баров = 72 часа
  
  // Параметры анализа
  minRuleSupport: number; // Минимальное количество срабатываний правила
  maxRuleConditions: number; // Максимум условий в комбинированном правиле
  
  // Walk-forward
  trainYears: number;
  testYears: number;
}

// Количество баров в 24 часах для разных таймфреймов
export const BARS_PER_24H: Record<AnalysisTimeframe, number> = {
  '15m': 96,
  '1h': 24,
  '4h': 6,
};

export const DEFAULT_CRASH_ANALYSIS_CONFIG: CrashAnalysisConfig = {
  // Настройки загрузки (максимум данных для качественного анализа)
  timeframe: '15m', // 15m — максимальная детализация
  periodYears: 5, // 5 лет — максимум данных
  
  // Оптимизированные параметры для предсказания crash'ей
  crashThresholdPct: 5, // 5% падение
  crashWindowBars: 24, // 6 часов при 15m таймфрейме
  zscore24hBars: 96, // 96 баров = 24 часа при 15m
  zscore72hBars: 288, // 288 баров = 72 часа при 15m
  minRuleSupport: 20, // Больше поддержки = надёжнее правила
  maxRuleConditions: 5,
  trainYears: 3, // 3 года обучения
  testYears: 1,
};

// ═══════════════════════════════════════════════════════════════════
// ПРОГРЕСС ЗАГРУЗКИ
// ═══════════════════════════════════════════════════════════════════

export interface DataLoadProgress {
  stage: 'idle' | 'loading-ohlcv' | 'loading-oi' | 'loading-funding' | 'loading-feargreed' | 'loading-spot' | 'processing' | 'analyzing' | 'ml-training' | 'done' | 'error';
  progress: number; // 0-100
  message: string;
  error?: string;
}
