(() => {
  const BACKGROUND_SOURCE = 'veles-background';
  const CONTENT_SOURCE = 'veles-content';
  const PAGE_SOURCE = 'veles-bridge';
  const INJECTED_SOURCE = 'veles-injected';

  const pendingRequestIds = new Set();
  let injected = false;

  // The background re-injects this script into tabs opened before an extension update. When several copies end up
  // in one page, only the newest handles new requests (otherwise a POST would be sent twice); older copies still
  // deliver answers to the requests they already accepted.
  const GENERATION_KEY = '__velesProxyBridgeGeneration';
  const generation = (Number(globalThis[GENERATION_KEY]) || 0) + 1;
  globalThis[GENERATION_KEY] = generation;
  const isCurrentGeneration = () => globalThis[GENERATION_KEY] === generation;

  const isExtensionAlive = () => {
    try {
      return Boolean(chrome.runtime?.id);
    } catch {
      return false;
    }
  };

  const sendToBackground = (message) => {
    if (!isExtensionAlive()) {
      return;
    }
    try {
      const result = chrome.runtime.sendMessage(message);
      if (result && typeof result.catch === 'function') {
        result.catch(() => undefined);
      }
    } catch (error) {
      console.warn('[Veles proxy bridge] Unable to reach the extension', error);
    }
  };

  // Генерируем nonce для handshake
  const BRIDGE_NONCE =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

  const injectPageBridge = () => {
    if (injected) {
      return;
    }

    try {
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL('scripts/page-fetch-bridge.js');
      script.async = false;
      // передаём nonce инжектируемому скрипту через data атрибут
      script.dataset.velesBridgeNonce = BRIDGE_NONCE;
      script.addEventListener('load', () => {
        script.remove();
        // The page script has run by now (load fires after execution): deliver what was waiting for it.
        markBridgeReady();
      });
      script.addEventListener('error', () => {
        script.remove();
        injected = false;
      });

      (document.head || document.documentElement).appendChild(script);
      injected = true;
    } catch (error) {
      console.error('[Veles proxy bridge] Unable to inject page bridge', error);
    }
  };

  // Messages posted before the page bridge is listening would be lost (the first request after a page load used
  // to hang until the timeout) — keep them until it is ready.
  let bridgeReady = false;
  const outbox = [];

  const markBridgeReady = () => {
    if (bridgeReady) {
      return;
    }
    bridgeReady = true;
    for (const message of outbox.splice(0)) {
      window.postMessage(message, window.location.origin);
    }
  };

  const postToPage = (message) => {
    if (!bridgeReady) {
      outbox.push(message);
      injectPageBridge();
      return;
    }
    window.postMessage(message, window.location.origin);
  };

  const forwardToPage = (requestId, payload) => {
    postToPage({
      source: PAGE_SOURCE,
      action: 'proxy-request',
      requestId,
      payload,
      nonce: BRIDGE_NONCE,
    });
  };

  window.addEventListener('message', (event) => {
    if (event.source !== window) {
      return;
    }

    const data = event.data;
    if (!data || typeof data !== 'object') {
      return;
    }

    if (data.source === INJECTED_SOURCE && data.action === 'proxy-response') {
      const { requestId, payload, nonce } = data;
      if (nonce !== BRIDGE_NONCE) {
        console.warn('[Veles proxy bridge] Ignored proxy-response with invalid nonce');
        return;
      }
      if (!pendingRequestIds.has(requestId)) {
        return;
      }

      pendingRequestIds.delete(requestId);

      sendToBackground({
        source: CONTENT_SOURCE,
        action: 'proxy-response',
        requestId,
        payload,
      });
      return;
    }

    if (data.source === INJECTED_SOURCE && data.action === 'ping-response') {
      const { payload, nonce } = data;
      if (nonce !== BRIDGE_NONCE) {
        console.warn('[Veles proxy bridge] Ignored ping-response with invalid nonce');
        return;
      }
      if (!isCurrentGeneration()) {
        return;
      }
      sendToBackground({
        source: CONTENT_SOURCE,
        action: 'ping-response',
        payload,
      });
    }
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || typeof message !== 'object') {
      return false;
    }

    if (!isCurrentGeneration()) {
      return false;
    }

    if (message.source === BACKGROUND_SOURCE && message.action === 'proxy-request') {
      const { requestId, payload } = message;
      if (!(requestId && payload)) {
        sendResponse({
          accepted: false,
          error: 'Некорректный формат запроса.',
        });
        return false;
      }

      injectPageBridge();
      pendingRequestIds.add(requestId);
      forwardToPage(requestId, payload);

      sendResponse({ accepted: true });
      return false;
    }

    if (message.source === BACKGROUND_SOURCE && message.action === 'ping') {
      injectPageBridge();
      postToPage({
        source: PAGE_SOURCE,
        action: 'ping',
        nonce: BRIDGE_NONCE,
      });
      sendResponse({ accepted: true });
      return false;
    }

    return false;
  });

  // Prepare the page bridge right away so the first request does not wait for it.
  injectPageBridge();

  console.info('[Veles proxy bridge] готов');
})();
