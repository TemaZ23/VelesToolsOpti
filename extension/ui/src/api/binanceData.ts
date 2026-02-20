/**
 * API для получения исторических данных BTC с Binance
 * 
 * Доступные бесплатные данные:
 * - OHLCV (Klines) - полная история
 * - Funding Rate - история с 2019
 * - Open Interest - ограниченная история (~30 дней через публичный API)
 * 
 * Недоступные данные (требуют платные API):
 * - Глубина стакана (order book) - только текущий snapshot
 * - Ликвидации - нет публичного исторического API
 * - On-chain метрики - требуют Glassnode/CryptoQuant
 */

import type { FundingRateBar, OpenInterestBar, RawBar } from '../types/crashAnalysis';

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

const BINANCE_FUTURES_BASE = 'https://fapi.binance.com';
const BINANCE_SPOT_BASE = 'https://api.binance.com';

// Лимиты Binance API
const KLINES_LIMIT = 1500; // Макс за 1 запрос
const FUNDING_LIMIT = 1000;
const OI_LIMIT = 500;

// Задержка между запросами для избежания rate limit
const REQUEST_DELAY_MS = 200;

// ═══════════════════════════════════════════════════════════════════
// УТИЛИТЫ
// ═══════════════════════════════════════════════════════════════════

const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Безопасный fetch с retry
 */
const fetchWithRetry = async (url: string, retries = 3): Promise<Response> => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return response;
      }
      if (response.status === 429) {
        // Rate limited - wait longer
        await delay(5000);
        continue;
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    } catch (error) {
      if (i === retries - 1) throw error;
      await delay(1000 * (i + 1));
    }
  }
  throw new Error('Max retries exceeded');
};

// ═══════════════════════════════════════════════════════════════════
// OHLCV (KLINES)
// ═══════════════════════════════════════════════════════════════════

/**
 * Загрузка OHLCV данных с Binance Futures
 */
export const fetchKlines = async (
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number,
  onProgress?: (loaded: number, total: number) => void,
): Promise<RawBar[]> => {
  const bars: RawBar[] = [];
  let currentStart = startTime;
  const totalMs = endTime - startTime;
  
  while (currentStart < endTime) {
    const url = `${BINANCE_FUTURES_BASE}/fapi/v1/klines?symbol=${symbol}&interval=${interval}&startTime=${currentStart}&endTime=${endTime}&limit=${KLINES_LIMIT}`;
    
    const response = await fetchWithRetry(url);
    const data: BinanceKline[] = await response.json();
    
    if (data.length === 0) break;
    
    for (const k of data) {
      const takerBuyVolume = parseFloat(k[9]);
      const totalVolume = parseFloat(k[5]);
      
      bars.push({
        timestamp: k[0],
        open: parseFloat(k[1]),
        high: parseFloat(k[2]),
        low: parseFloat(k[3]),
        close: parseFloat(k[4]),
        volume: totalVolume,
        quoteVolume: parseFloat(k[7]),
        takerBuyVolume,
        takerSellVolume: totalVolume - takerBuyVolume,
      });
    }
    
    // Следующий запрос с последнего timestamp + 1
    currentStart = data[data.length - 1][0] + 1;
    
    if (onProgress) {
      const progress = Math.min(100, ((currentStart - startTime) / totalMs) * 100);
      onProgress(bars.length, Math.round(progress));
    }
    
    await delay(REQUEST_DELAY_MS);
  }
  
  return bars;
};

// ═══════════════════════════════════════════════════════════════════
// FUNDING RATE
// ═══════════════════════════════════════════════════════════════════

interface BinanceFundingRate {
  symbol: string;
  fundingTime: number;
  fundingRate: string;
  markPrice: string;
}

/**
 * Загрузка Funding Rate с Binance Futures
 * Funding происходит каждые 8 часов
 */
export const fetchFundingRates = async (
  symbol: string,
  startTime: number,
  endTime: number,
  onProgress?: (loaded: number) => void,
): Promise<FundingRateBar[]> => {
  const rates: FundingRateBar[] = [];
  let currentStart = startTime;
  
  while (currentStart < endTime) {
    const url = `${BINANCE_FUTURES_BASE}/fapi/v1/fundingRate?symbol=${symbol}&startTime=${currentStart}&endTime=${endTime}&limit=${FUNDING_LIMIT}`;
    
    const response = await fetchWithRetry(url);
    const data: BinanceFundingRate[] = await response.json();
    
    if (data.length === 0) break;
    
    for (const f of data) {
      rates.push({
        timestamp: f.fundingTime,
        fundingRate: parseFloat(f.fundingRate),
      });
    }
    
    currentStart = data[data.length - 1].fundingTime + 1;
    
    if (onProgress) {
      onProgress(rates.length);
    }
    
    await delay(REQUEST_DELAY_MS);
  }
  
  return rates;
};

// ═══════════════════════════════════════════════════════════════════
// OPEN INTEREST
// ═══════════════════════════════════════════════════════════════════

interface BinanceOpenInterest {
  symbol: string;
  sumOpenInterest: string;
  sumOpenInterestValue: string;
  timestamp: number;
}

/**
 * Загрузка Open Interest с Binance Futures
 * ВНИМАНИЕ: Публичный API даёт только ~30 дней истории!
 * Для полной истории нужен Data API (платный)
 */
export const fetchOpenInterest = async (
  symbol: string,
  period: '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '12h' | '1d',
  startTime: number,
  endTime: number,
  onProgress?: (loaded: number) => void,
): Promise<OpenInterestBar[]> => {
  const oi: OpenInterestBar[] = [];
  let currentStart = startTime;
  
  while (currentStart < endTime) {
    const url = `${BINANCE_FUTURES_BASE}/futures/data/openInterestHist?symbol=${symbol}&period=${period}&startTime=${currentStart}&endTime=${endTime}&limit=${OI_LIMIT}`;
    
    try {
      const response = await fetchWithRetry(url);
      const data: BinanceOpenInterest[] = await response.json();
      
      if (data.length === 0) break;
      
      for (const o of data) {
        oi.push({
          timestamp: o.timestamp,
          openInterest: parseFloat(o.sumOpenInterest),
          openInterestValue: parseFloat(o.sumOpenInterestValue),
        });
      }
      
      currentStart = data[data.length - 1].timestamp + 1;
      
      if (onProgress) {
        onProgress(oi.length);
      }
    } catch (error) {
      // OI history may be limited, just break
      console.warn('[fetchOpenInterest] Error or limit reached:', error);
      break;
    }
    
    await delay(REQUEST_DELAY_MS);
  }
  
  return oi;
};

// ═══════════════════════════════════════════════════════════════════
// TAKER BUY/SELL RATIO (альтернативный источник)
// ═══════════════════════════════════════════════════════════════════

interface BinanceTakerRatio {
  buySellRatio: string;
  buyVol: string;
  sellVol: string;
  timestamp: number;
}

/**
 * Загрузка Taker Buy/Sell Ratio
 * Может быть полезно как дополнение к данным из klines
 */
export const fetchTakerBuySellRatio = async (
  symbol: string,
  period: '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '12h' | '1d',
  startTime: number,
  endTime: number,
  onProgress?: (loaded: number) => void,
): Promise<Array<{ timestamp: number; buyVol: number; sellVol: number; ratio: number }>> => {
  const ratios: Array<{ timestamp: number; buyVol: number; sellVol: number; ratio: number }> = [];
  let currentStart = startTime;
  
  while (currentStart < endTime) {
    const url = `${BINANCE_FUTURES_BASE}/futures/data/takerlongshortRatio?symbol=${symbol}&period=${period}&startTime=${currentStart}&endTime=${endTime}&limit=${OI_LIMIT}`;
    
    try {
      const response = await fetchWithRetry(url);
      const data: BinanceTakerRatio[] = await response.json();
      
      if (data.length === 0) break;
      
      for (const r of data) {
        ratios.push({
          timestamp: r.timestamp,
          buyVol: parseFloat(r.buyVol),
          sellVol: parseFloat(r.sellVol),
          ratio: parseFloat(r.buySellRatio),
        });
      }
      
      currentStart = data[data.length - 1].timestamp + 1;
      
      if (onProgress) {
        onProgress(ratios.length);
      }
    } catch (error) {
      console.warn('[fetchTakerBuySellRatio] Error:', error);
      break;
    }
    
    await delay(REQUEST_DELAY_MS);
  }
  
  return ratios;
};

// ═══════════════════════════════════════════════════════════════════
// LONG/SHORT RATIO (Top Traders)
// ═══════════════════════════════════════════════════════════════════

interface BinanceLongShortRatio {
  symbol: string;
  longShortRatio: string;
  longAccount: string;
  shortAccount: string;
  timestamp: number;
}

/**
 * Загрузка Long/Short Ratio топ трейдеров
 */
export const fetchTopTraderLongShortRatio = async (
  symbol: string,
  period: '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '12h' | '1d',
  startTime: number,
  endTime: number,
  onProgress?: (loaded: number) => void,
): Promise<Array<{ timestamp: number; longRatio: number; shortRatio: number }>> => {
  const ratios: Array<{ timestamp: number; longRatio: number; shortRatio: number }> = [];
  let currentStart = startTime;
  
  while (currentStart < endTime) {
    const url = `${BINANCE_FUTURES_BASE}/futures/data/topLongShortPositionRatio?symbol=${symbol}&period=${period}&startTime=${currentStart}&endTime=${endTime}&limit=${OI_LIMIT}`;
    
    try {
      const response = await fetchWithRetry(url);
      const data: BinanceLongShortRatio[] = await response.json();
      
      if (data.length === 0) break;
      
      for (const r of data) {
        ratios.push({
          timestamp: r.timestamp,
          longRatio: parseFloat(r.longAccount),
          shortRatio: parseFloat(r.shortAccount),
        });
      }
      
      currentStart = data[data.length - 1].timestamp + 1;
      
      if (onProgress) {
        onProgress(ratios.length);
      }
    } catch (error) {
      console.warn('[fetchTopTraderLongShortRatio] Error:', error);
      break;
    }
    
    await delay(REQUEST_DELAY_MS);
  }
  
  return ratios;
};

// ═══════════════════════════════════════════════════════════════════
// УТИЛИТЫ ДЛЯ ПЕРИОДОВ
// ═══════════════════════════════════════════════════════════════════

/**
 * Получить timestamp начала периода (N лет назад)
 */
export const getStartTimestamp = (yearsAgo: number): number => {
  const now = new Date();
  now.setFullYear(now.getFullYear() - yearsAgo);
  return now.getTime();
};

/**
 * Получить текущий timestamp
 */
export const getCurrentTimestamp = (): number => Date.now();

/**
 * Форматировать timestamp для отображения
 */
export const formatTimestamp = (ts: number): string => {
  return new Date(ts).toISOString().split('T')[0];
};

// ═══════════════════════════════════════════════════════════════════
// FEAR & GREED INDEX (Alternative.me)
// ═══════════════════════════════════════════════════════════════════

export interface FearGreedData {
  timestamp: number;
  value: number; // 0-100
  classification: string; // "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
}

/**
 * Загрузка Fear & Greed Index
 * Доступна полная история с 2018 года
 * Данные дневные (не 15m), будем интерполировать
 */
export const fetchFearGreedIndex = async (
  onProgress?: (loaded: number) => void,
): Promise<FearGreedData[]> => {
  try {
    // limit=0 возвращает всю историю
    const url = 'https://api.alternative.me/fng/?limit=0&format=json';
    const response = await fetchWithRetry(url);
    const json = await response.json();
    
    if (!json.data || !Array.isArray(json.data)) {
      console.warn('[fetchFearGreedIndex] Unexpected response format');
      return [];
    }
    
    const result: FearGreedData[] = json.data.map((item: { value: string; timestamp: string; value_classification: string }) => ({
      timestamp: parseInt(item.timestamp) * 1000, // API возвращает секунды
      value: parseInt(item.value),
      classification: item.value_classification,
    }));
    
    if (onProgress) {
      onProgress(result.length);
    }
    
    // Сортируем по времени (API возвращает в обратном порядке)
    return result.sort((a, b) => a.timestamp - b.timestamp);
  } catch (error) {
    console.warn('[fetchFearGreedIndex] Error:', error);
    return [];
  }
};

// ═══════════════════════════════════════════════════════════════════
// SPOT OHLCV (для расчёта Basis)
// ═══════════════════════════════════════════════════════════════════

/**
 * Загрузка Spot OHLCV для расчёта Spot-Futures Basis
 */
export const fetchSpotKlines = async (
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number,
  onProgress?: (loaded: number, pct: number) => void,
): Promise<RawBar[]> => {
  const bars: RawBar[] = [];
  let currentStart = startTime;
  const totalMs = endTime - startTime;
  
  while (currentStart < endTime) {
    const url = `${BINANCE_SPOT_BASE}/api/v3/klines?symbol=${symbol}&interval=${interval}&startTime=${currentStart}&endTime=${endTime}&limit=${KLINES_LIMIT}`;
    
    const response = await fetchWithRetry(url);
    const data: BinanceKline[] = await response.json();
    
    if (data.length === 0) break;
    
    for (const k of data) {
      const takerBuyVolume = parseFloat(k[9]);
      const totalVolume = parseFloat(k[5]);
      
      bars.push({
        timestamp: k[0],
        open: parseFloat(k[1]),
        high: parseFloat(k[2]),
        low: parseFloat(k[3]),
        close: parseFloat(k[4]),
        volume: totalVolume,
        quoteVolume: parseFloat(k[7]),
        takerBuyVolume,
        takerSellVolume: totalVolume - takerBuyVolume,
      });
    }
    
    currentStart = data[data.length - 1][0] + 1;
    
    if (onProgress) {
      const progress = Math.min(100, ((currentStart - startTime) / totalMs) * 100);
      onProgress(bars.length, Math.round(progress));
    }
    
    await delay(REQUEST_DELAY_MS);
  }
  
  return bars;
};

// Тип для Binance Kline (используется в обоих методах)
export interface BinanceKline {
  0: number;  // Open time
  1: string;  // Open
  2: string;  // High
  3: string;  // Low
  4: string;  // Close
  5: string;  // Volume
  6: number;  // Close time
  7: string;  // Quote volume
  8: number;  // Number of trades
  9: string;  // Taker buy base volume
  10: string; // Taker buy quote volume
  11: string; // Ignore
}
