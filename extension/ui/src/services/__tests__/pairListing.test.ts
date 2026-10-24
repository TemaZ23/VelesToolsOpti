import { beforeEach, describe, expect, it, vi } from 'vitest';

const { fetchPairsAvailabilityMock } = vi.hoisted(() => ({
  fetchPairsAvailabilityMock: vi.fn(),
}));

vi.mock('../../api/pairs', () => ({
  fetchPairsAvailability: fetchPairsAvailabilityMock,
}));

import { loadListingDates, normalizePairCode, pairCodeFor } from '../pairListing';

beforeEach(() => {
  fetchPairsAvailabilityMock.mockReset();
});

describe('pairListing', () => {
  it('нормализует код пары: только латиница и цифры в верхнем регистре', () => {
    expect(normalizePairCode('aave/usdt')).toBe('AAVEUSDT');
    expect(normalizePairCode('1000PEPE-USDT')).toBe('1000PEPEUSDT');
    expect(pairCodeFor('aave', 'usdt')).toBe('AAVEUSDT');
  });

  it('собирает даты листинга по коду пары, пропуская записи без пары или с некорректной датой', async () => {
    fetchPairsAvailabilityMock.mockResolvedValue({
      payload: [
        { symbol: 'AAVE/USDT', availableFrom: '2020-10-15T00:00:00Z' },
        { symbol: 'NEWUSDT', availableFrom: '2025-03-01T00:00:00.000Z' },
        { symbol: null, availableFrom: '2020-01-01T00:00:00Z' },
        { symbol: 'BADUSDT', availableFrom: 'не дата' },
        { symbol: 'NODATEUSDT', availableFrom: null },
      ],
    });
    await expect(loadListingDates('BYBIT_FUTURES')).resolves.toEqual({
      AAVEUSDT: Date.parse('2020-10-15T00:00:00Z'),
      NEWUSDT: Date.parse('2025-03-01T00:00:00.000Z'),
    });
    expect(fetchPairsAvailabilityMock).toHaveBeenCalledWith('BYBIT_FUTURES');
  });

  it('пустой словарь — пустой результат', async () => {
    fetchPairsAvailabilityMock.mockResolvedValue({ payload: null });
    await expect(loadListingDates('BINANCE_FUTURES')).resolves.toEqual({});
  });
});
