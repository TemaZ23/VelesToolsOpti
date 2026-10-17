/**
 * Даты листинга пар биржи (словарь доступности Veles) для портфельного оптимизатора:
 * бэктест монеты начинается не раньше её листинга, а монеты, которые в выбранный период
 * ещё не торговались, исключаются из корзины (как делает Veles Tools 2.15).
 */

import { fetchPairsAvailability } from '../api/pairs';
import { toTimestamp } from '../lib/dateTime';

/** Код пары для сравнения: только латиница и цифры в верхнем регистре («aave/usdt» → «AAVEUSDT»). */
export const normalizePairCode = (value: string): string => value.replace(/[^A-Za-z0-9]/g, '').toUpperCase();

/** Код пары монеты корзины: тикер + валюта котировки. */
export const pairCodeFor = (ticker: string, quoteCurrency: string): string =>
  normalizePairCode(`${ticker}${quoteCurrency}`);

/** Даты листинга пар биржи: код пары → мс epoch. Пары без пары или с некорректной датой пропускаются. */
export const loadListingDates = async (exchange: string): Promise<Record<string, number>> => {
  const dictionary = await fetchPairsAvailability(exchange);
  const result: Record<string, number> = {};
  for (const item of Array.isArray(dictionary.payload) ? dictionary.payload : []) {
    if (!(item && typeof item.symbol === 'string')) {
      continue;
    }
    const code = normalizePairCode(item.symbol);
    const listedAt = toTimestamp(item.availableFrom);
    if (code && listedAt !== null) {
      result[code] = listedAt;
    }
  }
  return result;
};
