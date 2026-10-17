const UI_MESSAGE_SOURCE = 'veles-ui';
const CONTENT_MESSAGE_SOURCE = 'veles-content';
const BACKGROUND_MESSAGE_SOURCE = 'veles-background';

/** How long a dispatched request may wait for the tab before it is retried (GET) or failed (POST/PUT/DELETE). */
const REQUEST_TIMEOUT_MS = 45000;
const MAX_ATTEMPTS = 4;
const RECONNECT_INTERVAL_MS = 2000;
/** A request that cannot be delivered for this long is answered with an error so the UI never hangs forever. */
const MAX_QUEUE_WAIT_MS = 120000;
const INJECT_COOLDOWN_MS = 4000;
const WAKE_COOLDOWN_MS = 20000;

const PROXY_BRIDGE_FILE = 'scripts/proxy-bridge.js';

const VELES_BASE_DOMAIN = 'veles.finance';

const VELES_URL_PATTERNS = [
  `https://${VELES_BASE_DOMAIN}/*`,
  `https://*.${VELES_BASE_DOMAIN}/*`,
  `http://${VELES_BASE_DOMAIN}/*`,
  `http://*.${VELES_BASE_DOMAIN}/*`,
];

/**
 * Errors meaning that nobody listens in the tab (content script not injected yet, page reloading, tab discarded
 * by Memory Saver…). The message was not processed, so it is safe to deliver it again.
 */
const MISSING_RECEIVER_PATTERNS = [
  'receiving end does not exist',
  'could not establish connection',
  'message port closed',
  'frame with id 0',
  'no frame with id',
];

const pendingRequests = new Map();
const requestQueue = [];

let requestDelayMs = 400;
let activeRequest = null;

let lastPing = 0;
let lastPingResult = { ok: false, error: 'Нет данных о соединении' };
let isConnected = true;
let reconnectTimer = null;
let activeOrigin = null;
/** The tab that answered last: requests stick to it instead of jumping between several veles tabs. */
let preferredTabId = null;

const injectAttempts = new Map();
const wakeAttempts = new Map();
const protectedTabs = new Set();

const errorMessageOf = (error) => (error instanceof Error ? error.message : String(error));

const isMissingReceiverError = (message) => {
  if (typeof message !== 'string') {
    return false;
  }
  const normalized = message.toLowerCase();
  return MISSING_RECEIVER_PATTERNS.some((pattern) => normalized.includes(pattern));
};

const isVelesHostname = (hostname) => {
  if (typeof hostname !== 'string' || hostname.length === 0) {
    return false;
  }
  const normalized = hostname.toLowerCase();
  return normalized === VELES_BASE_DOMAIN || normalized.endsWith(`.${VELES_BASE_DOMAIN}`);
};

const isVelesUrl = (url) => {
  if (!url || typeof url !== 'string') {
    return false;
  }
  try {
    const parsed = new URL(url);
    const isAllowedProtocol = parsed.protocol === 'https:' || parsed.protocol === 'http:';
    return isAllowedProtocol && isVelesHostname(parsed.hostname);
  } catch {
    return false;
  }
};

const updateActiveOrigin = (url) => {
  if (!isVelesUrl(url)) {
    return;
  }
  try {
    const parsed = new URL(url);
    activeOrigin = `${parsed.protocol}//${parsed.hostname}`;
  } catch (error) {
    console.warn('[Veles background] unable to parse origin from url', url, error);
  }
};

const normalizePayloadForOrigin = (payload) => {
  if (!payload || typeof payload.url !== 'string' || !activeOrigin) {
    return payload;
  }

  try {
    const targetOrigin = new URL(activeOrigin);
    const requestUrl = new URL(payload.url);

    if (!isVelesHostname(requestUrl.hostname)) {
      return payload;
    }

    requestUrl.protocol = targetOrigin.protocol;
    requestUrl.host = targetOrigin.host;

    return { ...payload, url: requestUrl.toString() };
  } catch (error) {
    console.warn('[Veles background] unable to normalize request payload for origin', error);
    return payload;
  }
};

const isIdempotentRequest = (payload) => {
  const method = typeof payload?.init?.method === 'string' ? payload.init.method.toUpperCase() : 'GET';
  return method === 'GET' || method === 'HEAD' || method === 'OPTIONS';
};

const broadcastConnectionStatus = () => {
  try {
    const response = chrome.runtime.sendMessage({
      source: BACKGROUND_MESSAGE_SOURCE,
      action: 'connection-status-update',
      payload: {
        ok: lastPingResult.ok,
        timestamp: lastPing,
        error: lastPingResult.error,
        origin: activeOrigin,
      },
    });

    if (response && typeof response.catch === 'function') {
      response.catch((error) => {
        if (isMissingReceiverError(errorMessageOf(error))) {
          return;
        }
        console.warn('[Veles background] Unable to broadcast connection status (async)', error);
      });
    }
  } catch (error) {
    console.warn('[Veles background] Unable to broadcast connection status', error);
  }
};

const clearReconnectTimer = () => {
  if (reconnectTimer) {
    clearInterval(reconnectTimer);
    reconnectTimer = null;
  }
};

const ensureReconnectLoop = () => {
  if (reconnectTimer) {
    return;
  }
  reconnectTimer = setInterval(() => {
    expireStaleRequests();
    attemptReconnect().catch((reason) => {
      console.warn('[Veles background] reconnect attempt failed:', reason);
    });
  }, RECONNECT_INTERVAL_MS);
};

const updateConnectionStatus = (ok, error) => {
  lastPing = Date.now();

  if (ok) {
    const changed = !lastPingResult.ok;
    lastPingResult = { ok: true };
    isConnected = true;
    clearReconnectTimer();
    if (changed) {
      broadcastConnectionStatus();
    }
    if (!activeRequest) {
      processQueue();
    }
    return;
  }

  lastPingResult = { ok: false, error: error ?? 'Нет соединения с вкладкой' };
  isConnected = false;
  broadcastConnectionStatus();
  ensureReconnectLoop();
};

const generateRequestId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

/** Memory Saver must not unload the tab the extension works through. */
const protectTab = (tab) => {
  if (!tab || typeof tab.id !== 'number' || protectedTabs.has(tab.id)) {
    return;
  }
  protectedTabs.add(tab.id);
  if (tab.autoDiscardable === false) {
    return;
  }
  try {
    const result = chrome.tabs.update(tab.id, { autoDiscardable: false });
    if (result && typeof result.catch === 'function') {
      result.catch(() => {
        protectedTabs.delete(tab.id);
      });
    }
  } catch {
    protectedTabs.delete(tab.id);
  }
};

/** A discarded tab has no scripts at all: reload it so the bridge comes back. */
const wakeTab = async (tab) => {
  if (!tab?.discarded || typeof tab.id !== 'number') {
    return;
  }
  const last = wakeAttempts.get(tab.id) ?? 0;
  if (Date.now() - last < WAKE_COOLDOWN_MS) {
    return;
  }
  wakeAttempts.set(tab.id, Date.now());
  try {
    await chrome.tabs.reload(tab.id);
  } catch (error) {
    console.warn('[Veles background] unable to reload discarded tab', error);
  }
};

/**
 * Injects the content script into a tab that was opened before the extension was installed/updated
 * (Chrome does not inject content scripts into already open tabs) — no need to refresh veles.finance by hand.
 */
const injectBridge = async (tabId) => {
  const last = injectAttempts.get(tabId) ?? 0;
  if (Date.now() - last < INJECT_COOLDOWN_MS) {
    return false;
  }
  injectAttempts.set(tabId, Date.now());
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: [PROXY_BRIDGE_FILE] });
    return true;
  } catch (error) {
    console.warn('[Veles background] unable to inject proxy bridge', error);
    return false;
  }
};

const tabScore = (tab) => {
  let score = 0;
  if (tab.id === preferredTabId) {
    score += 16;
  }
  if (!tab.discarded) {
    score += 8;
  }
  if (tab.status === 'complete') {
    score += 4;
  }
  if (tab.active) {
    score += 2;
  }
  return score;
};

const listVelesTabs = async () => {
  const tabs = await chrome.tabs.query({ url: VELES_URL_PATTERNS });
  return (tabs ?? []).filter((tab) => typeof tab.id === 'number' && isVelesUrl(tab.url));
};

const findActiveVelesTab = async () => {
  const candidates = await listVelesTabs();
  if (candidates.length === 0) {
    preferredTabId = null;
    return null;
  }

  const [tab] = [...candidates].sort((a, b) => {
    const diff = tabScore(b) - tabScore(a);
    return diff !== 0 ? diff : (b.lastAccessed ?? 0) - (a.lastAccessed ?? 0);
  });
  if (tab.url) {
    updateActiveOrigin(tab.url);
  }
  protectTab(tab);
  if (tab.discarded) {
    await wakeTab(tab);
  }
  return tab.id;
};

const sendToTab = (tabId, message) =>
  new Promise((resolve) => {
    try {
      chrome.tabs.sendMessage(tabId, message, { frameId: 0 }, (response) => {
        const lastError = chrome.runtime.lastError;
        if (lastError) {
          resolve({ error: lastError.message ?? 'Неизвестная ошибка вкладки' });
          return;
        }
        resolve({ response });
      });
    } catch (error) {
      resolve({ error: errorMessageOf(error) });
    }
  });

/** Sends a message to the tab; when no content script listens there, injects it and tries once more. */
const sendToTabWithRecovery = async (tabId, message) => {
  const first = await sendToTab(tabId, message);
  if (!(first.error && isMissingReceiverError(first.error))) {
    return first;
  }
  const injected = await injectBridge(tabId);
  if (!injected) {
    return first;
  }
  return sendToTab(tabId, message);
};

const createQueueItem = (requestId, payload, sendResponse, attempts = 0, createdAt = Date.now()) => ({
  requestId,
  payload,
  sendResponse,
  attempts,
  createdAt,
});

const resetActiveRequest = () => {
  if (activeRequest?.timeoutId) {
    clearTimeout(activeRequest.timeoutId);
  }
  activeRequest = null;
};

const scheduleNext = () => {
  setTimeout(processQueue, requestDelayMs);
};

const respondSafely = (sendResponse, payload) => {
  try {
    sendResponse(payload);
  } catch (error) {
    console.warn('[Veles background] unable to respond to UI', error);
  }
};

const failRequestImmediately = (sendResponse, requestId, message) => {
  respondSafely(sendResponse, { requestId, ok: false, error: message });
  scheduleNext();
};

/** Answers requests that could not be delivered for too long, so the UI can decide what to do. */
const expireStaleRequests = () => {
  const now = Date.now();
  for (let index = requestQueue.length - 1; index >= 0; index--) {
    const item = requestQueue[index];
    if (now - item.createdAt > MAX_QUEUE_WAIT_MS) {
      requestQueue.splice(index, 1);
      respondSafely(item.sendResponse, {
        requestId: item.requestId,
        ok: false,
        error: `Нет связи с вкладкой ${VELES_BASE_DOMAIN}: ${lastPingResult.error ?? 'вкладка не отвечает'}`,
      });
    }
  }
};

const retryActiveRequest = (message, { incrementAttempt = true } = {}) => {
  if (!activeRequest) {
    return;
  }

  const { requestId, payload, sendResponse, attempts, createdAt } = activeRequest;
  pendingRequests.delete(requestId);
  resetActiveRequest();

  const nextAttempts = incrementAttempt ? attempts + 1 : attempts;
  if (incrementAttempt && nextAttempts > MAX_ATTEMPTS) {
    failRequestImmediately(sendResponse, requestId, message);
    return;
  }
  if (Date.now() - createdAt > MAX_QUEUE_WAIT_MS) {
    failRequestImmediately(sendResponse, requestId, `Нет связи с вкладкой ${VELES_BASE_DOMAIN}: ${message}`);
    return;
  }

  requestQueue.unshift(createQueueItem(requestId, payload, sendResponse, nextAttempts, createdAt));
  if (message) {
    updateConnectionStatus(false, message);
  }
  scheduleNext();
};

const finalizeRequest = (requestId, sendResponse, payload) => {
  resetActiveRequest();
  respondSafely(sendResponse, { requestId, ...payload });
  scheduleNext();
};

const markTabAlive = (tabId) => {
  if (typeof tabId === 'number') {
    preferredTabId = tabId;
  }
};

const pingTab = async (tabId) => {
  const { response, error } = await sendToTabWithRecovery(tabId, {
    source: BACKGROUND_MESSAGE_SOURCE,
    action: 'ping',
  });
  if (error) {
    return { ok: false, error };
  }
  if (response && (response.ok || response.accepted)) {
    markTabAlive(tabId);
    return { ok: true };
  }
  return { ok: false, error: response?.error ?? 'Нет ответа от контента' };
};

const attemptReconnect = async () => {
  const tabId = await findActiveVelesTab();
  if (!tabId) {
    updateConnectionStatus(false, `Не найдена вкладка ${VELES_BASE_DOMAIN}`);
    return;
  }

  const result = await pingTab(tabId);
  updateConnectionStatus(result.ok, result.error);
};

const processQueue = () => {
  if (activeRequest || requestQueue.length === 0) {
    return;
  }

  if (!isConnected) {
    ensureReconnectLoop();
    return;
  }

  const queueItem = requestQueue.shift();
  if (!queueItem) {
    return;
  }

  findActiveVelesTab()
    .then((tabId) => {
      if (!tabId) {
        requestQueue.unshift(queueItem);
        updateConnectionStatus(false, `Не найдена вкладка ${VELES_BASE_DOMAIN}`);
        return;
      }

      dispatchToTab(tabId, queueItem);
    })
    .catch((error) => {
      requestQueue.unshift(queueItem);
      updateConnectionStatus(false, errorMessageOf(error));
    });
};

const dispatchToTab = (tabId, queueItem) => {
  const { requestId, payload, sendResponse, attempts, createdAt } = queueItem;

  const normalizedPayload = normalizePayloadForOrigin(payload);

  activeRequest = {
    requestId,
    payload: normalizedPayload,
    sendResponse,
    attempts,
    createdAt,
    tabId,
    accepted: false,
    timeoutId: null,
  };

  const timeoutId = setTimeout(() => {
    if (activeRequest?.requestId !== requestId) {
      return;
    }
    if (activeRequest.accepted && !isIdempotentRequest(normalizedPayload)) {
      // The page may have executed it already — repeating a POST could start a second backtest.
      pendingRequests.delete(requestId);
      finalizeRequest(requestId, sendResponse, {
        ok: false,
        error: 'Таймаут ответа от вкладки (запрос мог выполниться на стороне veles.finance)',
      });
      return;
    }
    retryActiveRequest('Таймаут ответа от вкладки', { incrementAttempt: true });
  }, REQUEST_TIMEOUT_MS);
  activeRequest.timeoutId = timeoutId;

  pendingRequests.set(requestId, (responsePayload) => {
    clearTimeout(timeoutId);
    pendingRequests.delete(requestId);
    markTabAlive(tabId);
    updateConnectionStatus(true);
    finalizeRequest(requestId, sendResponse, responsePayload);
  });

  sendToTabWithRecovery(tabId, {
    source: BACKGROUND_MESSAGE_SOURCE,
    action: 'proxy-request',
    requestId,
    payload: normalizedPayload,
  }).then(({ response, error }) => {
    if (activeRequest?.requestId !== requestId) {
      return;
    }
    if (error) {
      // Nobody received the request — it is safe to deliver it again once the tab is back.
      retryActiveRequest(error, { incrementAttempt: !isMissingReceiverError(error) });
      return;
    }

    if (response && response.accepted === false) {
      retryActiveRequest(response.error ?? 'Контент-скрипт отклонил запрос.', { incrementAttempt: true });
      return;
    }

    activeRequest.accepted = true;
    markTabAlive(tabId);
    updateConnectionStatus(true);
  });
};

/** On install/update/browser start: bring the bridge into veles tabs that are already open. */
const injectIntoOpenTabs = async () => {
  try {
    const tabs = await listVelesTabs();
    for (const tab of tabs) {
      protectTab(tab);
      if (tab.discarded) {
        continue;
      }
      const result = await sendToTab(tab.id, { source: BACKGROUND_MESSAGE_SOURCE, action: 'ping' });
      if (result.error && isMissingReceiverError(result.error)) {
        await injectBridge(tab.id);
      }
    }
  } catch (error) {
    console.warn('[Veles background] unable to prepare open veles tabs', error);
  }
};

chrome.runtime.onInstalled.addListener(() => {
  injectIntoOpenTabs();
});

chrome.runtime.onStartup.addListener(() => {
  injectIntoOpenTabs();
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message !== 'object') {
    return false;
  }

  if (message.source === UI_MESSAGE_SOURCE) {
    const senderUrl = _sender?.url ?? '';
    const trustedPrefix = chrome.runtime.getURL('ui/dist/');
    if (!senderUrl?.startsWith(trustedPrefix)) {
      sendResponse({ ok: false, error: 'Unauthorized sender' });
      return false;
    }
  }

  if (message.source === UI_MESSAGE_SOURCE && message.action === 'ping') {
    findActiveVelesTab()
      .then(async (tabId) => {
        if (!tabId) {
          updateConnectionStatus(false, `Не найдена вкладка ${VELES_BASE_DOMAIN}`);
          sendResponse({
            ok: false,
            error: `Не найдена вкладка ${VELES_BASE_DOMAIN}`,
          });
          return;
        }

        const result = await pingTab(tabId);
        updateConnectionStatus(result.ok, result.error);
        sendResponse(result.ok ? { ok: true, origin: activeOrigin } : { ok: false, error: result.error });
      })
      .catch((error) => {
        const messageText = errorMessageOf(error);
        updateConnectionStatus(false, messageText);
        sendResponse({ ok: false, error: messageText });
      });

    return true;
  }

  if (message.source === UI_MESSAGE_SOURCE && message.action === 'keepalive') {
    // Keeps the service worker alive during long optimizations and restores settings after its restart.
    const delayCandidate = Number(message.payload?.delayMs);
    if (Number.isFinite(delayCandidate) && delayCandidate >= 0) {
      requestDelayMs = delayCandidate;
    }
    if (!isConnected) {
      ensureReconnectLoop();
    } else if (!activeRequest && requestQueue.length > 0) {
      processQueue();
    }
    sendResponse({
      ok: lastPingResult.ok,
      timestamp: lastPing,
      error: lastPingResult.error,
      origin: activeOrigin,
      queued: requestQueue.length,
    });
    return false;
  }

  if (message.source === UI_MESSAGE_SOURCE && message.action === 'active-veles-tab') {
    chrome.tabs
      .query({ url: VELES_URL_PATTERNS })
      .then((tabs) => {
        const sorted = [...(tabs ?? [])].sort((a, b) => {
          if (a.active !== b.active) {
            return a.active ? -1 : 1;
          }
          return (b.lastAccessed ?? 0) - (a.lastAccessed ?? 0);
        });
        sendResponse({
          ok: true,
          tabs: sorted.map((tab) => ({
            url: tab.url ?? null,
            active: Boolean(tab.active),
            lastAccessed: tab.lastAccessed ?? 0,
          })),
        });
      })
      .catch((error) => {
        sendResponse({ ok: false, error: errorMessageOf(error), tabs: [] });
      });
    return true;
  }

  if (message.source === UI_MESSAGE_SOURCE && message.action === 'connection-status') {
    sendResponse({
      ok: lastPingResult.ok,
      timestamp: lastPing,
      error: lastPingResult.error,
      origin: activeOrigin,
    });
    return false;
  }

  if (message.source === UI_MESSAGE_SOURCE && message.action === 'update-delay') {
    const delayCandidate = Number(message.payload?.delayMs);
    if (!Number.isFinite(delayCandidate) || delayCandidate < 0) {
      sendResponse({ ok: false, error: 'Некорректное значение задержки.' });
      return false;
    }

    requestDelayMs = delayCandidate;
    sendResponse({ ok: true, delayMs: requestDelayMs });
    return false;
  }

  if (message.source === UI_MESSAGE_SOURCE && message.action === 'proxy-request') {
    const payload = message.payload;
    if (!payload || typeof payload.url !== 'string') {
      sendResponse({ ok: false, error: 'Некорректный payload запроса.' });
      return false;
    }

    const requestId = generateRequestId();
    requestQueue.push(createQueueItem(requestId, payload, sendResponse));
    processQueue();
    return true;
  }

  if (message.source === CONTENT_MESSAGE_SOURCE && message.action === 'ping-response') {
    if (message.payload?.ok) {
      if (message.payload.origin) {
        updateActiveOrigin(message.payload.origin);
      }
      markTabAlive(_sender?.tab?.id);
      updateConnectionStatus(true);
    } else {
      updateConnectionStatus(false, message.payload?.error ?? 'Неизвестная ошибка');
    }
    return false;
  }

  if (message.source === CONTENT_MESSAGE_SOURCE && message.action === 'proxy-response') {
    const { requestId, payload } = message;
    const responder = pendingRequests.get(requestId);

    if (responder) {
      responder(payload);
    }

    return false;
  }

  return false;
});

chrome.tabs.onRemoved.addListener((tabId) => {
  protectedTabs.delete(tabId);
  injectAttempts.delete(tabId);
  wakeAttempts.delete(tabId);
  if (preferredTabId === tabId) {
    preferredTabId = null;
  }
  if (activeRequest && activeRequest.tabId === tabId) {
    retryActiveRequest('Вкладка закрыта.', { incrementAttempt: false });
  }

  findActiveVelesTab()
    .then((foundTabId) => {
      if (!foundTabId) {
        updateConnectionStatus(false, `Нет вкладки ${VELES_BASE_DOMAIN}`);
      }
    })
    .catch((error) => {
      updateConnectionStatus(false, errorMessageOf(error));
    });
});

chrome.tabs.onReplaced?.addListener((addedTabId, removedTabId) => {
  if (preferredTabId === removedTabId) {
    preferredTabId = addedTabId;
  }
  protectedTabs.delete(removedTabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.discarded === true && activeRequest && activeRequest.tabId === tabId) {
    retryActiveRequest('Вкладка выгружена браузером.', { incrementAttempt: false });
    return;
  }
  if (changeInfo.status === 'loading' || changeInfo.url) {
    const url = changeInfo.url ?? tab.url;
    if (url && !isVelesUrl(url)) {
      if (preferredTabId === tabId) {
        preferredTabId = null;
      }
      if (activeRequest && activeRequest.tabId === tabId) {
        retryActiveRequest(`Вкладка покинула ${VELES_BASE_DOMAIN}`, {
          incrementAttempt: false,
        });
      }
      findActiveVelesTab()
        .then((foundTabId) => {
          if (!foundTabId) {
            updateConnectionStatus(false, `Нет вкладки ${VELES_BASE_DOMAIN}`);
          }
        })
        .catch((error) => {
          updateConnectionStatus(false, errorMessageOf(error));
        });
    } else if (url) {
      // SPA navigation inside veles.finance keeps the page (and the request in flight) alive; a real reload is
      // handled by the request timeout and the automatic re-injection of the bridge.
      updateActiveOrigin(url);
    }
  }
  if (changeInfo.status === 'complete' && isVelesUrl(tab.url)) {
    protectTab(tab);
    if (!isConnected) {
      attemptReconnect().catch(() => undefined);
    }
  }
});
