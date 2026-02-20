export interface RuntimeMessage<TPayload = unknown> {
  source: string;
  action: string;
  payload?: TPayload;
}

export interface ProxyRequestPayload {
  url: string;
  init?: RequestInit;
}

export interface ProxyResponsePayload<TBody = unknown> {
  requestId: string;
  ok: boolean;
  status?: number;
  statusText?: string;
  body?: TBody;
  error?: string;
  headers?: Record<string, string>;
}

export interface ConnectionStatusSnapshot {
  ok: boolean;
  timestamp: number;
  error?: string;
  origin?: string | null;
}

export const isExtensionRuntime = (): boolean => {
  return typeof chrome !== 'undefined' && typeof chrome.runtime !== 'undefined' && Boolean(chrome.runtime.id);
};

/**
 * Проверка, является ли ошибка "потерей связи с расширением"
 */
const isConnectionLostError = (error: Error): boolean => {
  const message = error.message.toLowerCase();
  return (
    message.includes('receiving end does not exist') ||
    message.includes('extension context invalidated') ||
    message.includes('could not establish connection')
  );
};

/**
 * Задержка
 */
const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

export const sendRuntimeMessage = async <TResponse = unknown, TPayload = unknown>(
  message: RuntimeMessage<TPayload>,
  retries = 3,
): Promise<TResponse> => {
  if (!isExtensionRuntime()) {
    throw new Error('Расширение недоступно. Запустите Veles Tools как часть расширения.');
  }

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const result = await new Promise<TResponse>((resolve, reject) => {
        chrome.runtime.sendMessage(message, (response) => {
          const lastErr = chrome.runtime.lastError;
          if (lastErr) {
            reject(new Error(lastErr.message));
            return;
          }
          resolve(response as TResponse);
        });
      });
      return result;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      if (isConnectionLostError(lastError)) {
        if (attempt < retries) {
          console.warn(`[Extension] Попытка ${attempt}/${retries}: связь потеряна, повтор через 1с...`);
          await delay(1000);
          continue;
        }
        // После всех попыток - выдаём понятную ошибку
        throw new Error(
          'Связь с расширением потеряна. Расширение было обновлено или перезагружено. ' +
          'Пожалуйста, перезагрузите страницу (F5).'
        );
      }
      
      throw lastError;
    }
  }

  throw lastError ?? new Error('Неизвестная ошибка');
};

export const proxyHttpRequest = async <TBody = unknown>(
  payload: ProxyRequestPayload,
): Promise<ProxyResponsePayload<TBody>> => {
  const response = await sendRuntimeMessage<ProxyResponsePayload<TBody>>({
    source: 'veles-ui',
    action: 'proxy-request',
    payload,
  });

  return response;
};

export const pingConnection = async (): Promise<{
  ok: boolean;
  error?: string;
  origin?: string | null;
}> => {
  try {
    const response = await sendRuntimeMessage<{
      ok: boolean;
      error?: string;
      origin?: string | null;
    }>({
      source: 'veles-ui',
      action: 'ping',
    });
    return response;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, error: message };
  }
};

export const readConnectionStatus = async (): Promise<ConnectionStatusSnapshot> => {
  const response = await sendRuntimeMessage<ConnectionStatusSnapshot>({
    source: 'veles-ui',
    action: 'connection-status',
  });

  return response;
};

export const updateRequestDelay = async (delayMs: number): Promise<{ ok: boolean; delayMs: number }> => {
  const response = await sendRuntimeMessage<{ ok: boolean; delayMs: number; error?: string }, { delayMs: number }>({
    source: 'veles-ui',
    action: 'update-delay',
    payload: { delayMs },
  });

  if (!response.ok) {
    throw new Error(response.error ?? 'Не удалось обновить задержку запросов.');
  }

  return response;
};
