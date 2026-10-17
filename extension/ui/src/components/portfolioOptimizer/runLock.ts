/**
 * Защита от двух запусков портфельного оптимизатора в разных вкладках расширения (меню расширения
 * открывает новую вкладку на каждый клик, а состояние запуска хранится под одним ключом localStorage).
 *
 * Основной механизм — Web Locks: вкладка, в которой живёт оптимизатор, держит блокировку всё время его
 * работы, включая паузу; браузер освобождает её сам при закрытии или перезагрузке вкладки. Запасной
 * механизм (Web Locks недоступны) — «пульс»: такая вкладка раз в 30 с записывает отметку времени со
 * своим id, и другая вкладка считает запуск выполняющимся, пока отметка свежая.
 */

import { readStorageValue, removeStorageValue, writeStorageValue } from '../../lib/safeStorage';

/** Имя блокировки Web Locks на время жизни оптимизатора. */
export const PORTFOLIO_RUN_LOCK_NAME = 'veles-portfolio-optimizer';

/** Ключ localStorage с «пульсом» вкладки, в которой работает оптимизатор. */
export const PORTFOLIO_RUN_HEARTBEAT_KEY = 'veles_portfolio_optimizer_heartbeat';

/** Как часто вкладка с оптимизатором обновляет «пульс». */
export const RUN_HEARTBEAT_INTERVAL_MS = 30_000;

export interface RunLockHandle {
  release: () => void;
}

export interface RunLockProvider {
  /** Захват без ожидания: handle — блокировка получена, null — её держит другая вкладка. */
  acquire: () => Promise<RunLockHandle | null>;
  /** Держит ли блокировку какая-либо вкладка прямо сейчас. */
  isHeld: () => Promise<boolean>;
}

/** Отметка «оптимизатор работает в вкладке tabId» для запуска runId. */
export interface RunHeartbeat {
  runId: string;
  tabId: string;
  at: number;
}

/** Где ещё выполняется сохранённый запуск: 'locked' — точно (Web Locks), 'probable' — по «пульсу». */
export type OtherTabRun = 'locked' | 'probable' | 'none';

const readNavigatorLocks = (): LockManager | undefined =>
  typeof navigator === 'undefined' ? undefined : (navigator.locks ?? undefined);

/** Провайдер блокировки на Web Locks; null — API недоступно (используется «пульс»). */
export const createWebLockProvider = (
  locks: LockManager | undefined = readNavigatorLocks(),
): RunLockProvider | null => {
  if (!(locks && typeof locks.request === 'function' && typeof locks.query === 'function')) {
    return null;
  }
  return {
    acquire: () =>
      new Promise<RunLockHandle | null>((resolve, reject) => {
        let release: () => void = () => undefined;
        const held = new Promise<void>((done) => {
          release = done;
        });
        locks
          .request(PORTFOLIO_RUN_LOCK_NAME, { ifAvailable: true }, (lock) => {
            if (!lock) {
              resolve(null);
              return undefined;
            }
            resolve({ release });
            // Блокировка держится, пока не вызван release().
            return held;
          })
          .catch(reject);
      }),
    isHeld: async () => {
      const snapshot = await locks.query();
      return (snapshot.held ?? []).some((lock) => lock.name === PORTFOLIO_RUN_LOCK_NAME);
    },
  };
};

/** Случайный id вкладки для «пульса». */
export const createTabId = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
};

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

export const readRunHeartbeat = (): RunHeartbeat | null => {
  const raw = readStorageValue(PORTFOLIO_RUN_HEARTBEAT_KEY);
  if (!raw) {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (
      isRecord(parsed) &&
      typeof parsed.runId === 'string' &&
      typeof parsed.tabId === 'string' &&
      typeof parsed.at === 'number' &&
      Number.isFinite(parsed.at)
    ) {
      return { runId: parsed.runId, tabId: parsed.tabId, at: parsed.at };
    }
  } catch {
    // повреждённая отметка — как если бы её не было
  }
  return null;
};

export const writeRunHeartbeat = (heartbeat: RunHeartbeat): void => {
  writeStorageValue(PORTFOLIO_RUN_HEARTBEAT_KEY, JSON.stringify(heartbeat));
};

/** Удаляет «пульс», только если он принадлежит этой вкладке. */
export const clearRunHeartbeat = (tabId: string): void => {
  if (readRunHeartbeat()?.tabId === tabId) {
    removeStorageValue(PORTFOLIO_RUN_HEARTBEAT_KEY);
  }
};
