/** Пара биржи в словаре доступности Veles (GET /api/pairs/availability/dictionary). */
export interface PairAvailabilityDto {
  /** Пара, например «AAVEUSDT» или «AAVE/USDT». */
  symbol: string | null;
  /** С какого момента по паре есть данные (ISO). */
  availableFrom: string | null;
}

export interface PairAvailabilityDictionaryDto {
  payload: PairAvailabilityDto[] | null;
}
