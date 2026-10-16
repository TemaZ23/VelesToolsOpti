import { proxyHttpRequest } from '../lib/extensionMessaging';
import { resolveProxyErrorMessage } from '../lib/httpErrors';
import { buildApiUrl } from './baseUrl';
import type { PairAvailabilityDictionaryDto } from './pairs.dtos';

const PAIRS_AVAILABILITY_ENDPOINT = buildApiUrl('/api/pairs/availability/dictionary');

/** Словарь доступности пар биржи: с какой даты по каждой паре есть данные (дата листинга). */
export const fetchPairsAvailability = async (exchange: string): Promise<PairAvailabilityDictionaryDto> => {
  const query = new URLSearchParams({ exchange });
  const response = await proxyHttpRequest<PairAvailabilityDictionaryDto>({
    url: `${PAIRS_AVAILABILITY_ENDPOINT}?${query.toString()}`,
    init: {
      method: 'GET',
      credentials: 'include',
      headers: {
        accept: 'application/json, text/plain, */*',
      },
    },
  });

  if (!response.ok) {
    throw new Error(resolveProxyErrorMessage(response));
  }

  if (!response.body) {
    throw new Error('Пустой ответ сервера при загрузке дат листинга.');
  }

  return response.body;
};
