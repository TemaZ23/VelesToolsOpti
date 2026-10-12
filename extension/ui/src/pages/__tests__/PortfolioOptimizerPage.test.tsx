import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

const { fetchBotsMock, createBotMock, fetchBotStrategyMock } = vi.hoisted(() => ({
  fetchBotsMock: vi.fn(),
  createBotMock: vi.fn(),
  fetchBotStrategyMock: vi.fn(),
}));

vi.mock('../../api/bots', () => ({
  fetchBots: fetchBotsMock,
  createBot: createBotMock,
}));

vi.mock('../../api/backtestRunner', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/backtestRunner')>();
  return { ...actual, fetchBotStrategy: fetchBotStrategyMock };
});

import type { BotStrategy } from '../../api/backtestRunner';
import type { StrategyConditionDto } from '../../api/bots.dtos';
import { PORTFOLIO_FORM_STORAGE_KEY } from '../../components/portfolioOptimizer/formState';
import { portfolioRunSession } from '../../components/portfolioOptimizer/runSession';
import { BacktestGroupsProvider } from '../../context/BacktestGroupsContext';
import { extractParams } from '../../lib/botParams';
import {
  DEFAULT_OBJECTIVE,
  DEFAULT_SCHEDULER_CONFIG,
  DEFAULT_SEARCH_CONFIG,
  PORTFOLIO_OPTIMIZER_STATE_KEY,
} from '../../services/portfolioOptimizer';
import { readAggregationConfig, writeAggregationConfig } from '../../storage/backtestAggregationConfigStore';
import { readBacktestGroups, writeBacktestGroups } from '../../storage/backtestGroupsStore';
import type { Candidate, PortfolioMetrics, PortfolioOptimizerState } from '../../types/portfolioOptimizer';
import PortfolioOptimizerPage from '../PortfolioOptimizerPage';

// ═══════════════════════════════════════════════════════════════════
// ОКРУЖЕНИЕ
// ═══════════════════════════════════════════════════════════════════

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}

beforeAll(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  // jsdom не поддерживает псевдоэлементы в getComputedStyle (их запрашивает замер скроллбара таблицы antd).
  const getComputedStyle = window.getComputedStyle.bind(window);
  window.getComputedStyle = (element: Element) => getComputedStyle(element);
  if (!window.matchMedia) {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: (query: string): MediaQueryList => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }),
    });
  }
});

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ
// ═══════════════════════════════════════════════════════════════════

const condition = (indicator: string, value: number): StrategyConditionDto => ({
  type: 'INDICATOR',
  indicator,
  interval: 'ONE_HOUR',
  basic: false,
  value,
  operation: 'LESS',
  closed: true,
  reverse: false,
});

const createStrategy = (): BotStrategy => ({
  id: 77,
  name: 'LONG 22',
  symbols: ['ADA/USDT', 'XRP/USDT'],
  pair: { exchange: 'BYBIT_FUTURES', from: 'ADA', to: 'USDT', symbol: 'ADAUSDT' },
  exchange: 'BYBIT_FUTURES',
  algorithm: 'LONG',
  conditions: [condition('RSI', 40), condition('CCI', -100)],
  settings: {
    type: 'CUSTOM',
    baseOrder: null,
    orders: [
      { indent: 0, volume: 40, conditions: [] },
      { indent: 2, volume: 60, conditions: [] },
    ],
  },
  profit: { type: 'ALL', currency: 'USDT', checkPnl: 1, conditions: [] },
  deposit: { amount: 100, leverage: 10, marginType: 'CROSS', currency: 'USDT' },
  stopLoss: null,
  pullUp: null,
  portion: null,
});

const createSavedState = (): PortfolioOptimizerState => {
  const strategy = createStrategy();
  const params = extractParams(strategy);
  const values = Object.fromEntries(params.map((param) => [param.id, param.base]));
  return {
    version: 1,
    runId: 'posaved',
    config: {
      botId: 77,
      botName: 'LONG 22',
      basket: ['ADA', 'XRP'],
      screenBasket: ['ADA'],
      run: {
        periodFrom: '2022-01-01T00:00:00.000Z',
        periodTo: '2024-01-01T23:59:59.999Z',
        makerCommission: 0.02,
        takerCommission: 0.055,
        includeWicks: true,
      },
      block: 60,
      objective: { ...DEFAULT_OBJECTIVE },
      search: { ...DEFAULT_SEARCH_CONFIG, generations: 3 },
      scheduler: { ...DEFAULT_SCHEDULER_CONFIG },
      params,
      baseGroupId: null,
    },
    baseStrategy: strategy,
    quoteCurrency: 'USDT',
    baseCandidateId: 'posaved-base',
    bestCandidateId: 'posaved-base',
    candidates: [
      {
        id: 'posaved-base',
        hash: 'pbase',
        values,
        parentId: null,
        generation: 0,
        changed: [],
        mutated: [],
        screen: null,
        full: {
          stage: 'full',
          symbols: ['ADA', 'XRP'],
          backtestIds: { ADA: 11, XRP: 12 },
          failedSymbols: [],
          status: 'done',
          metrics: {
            totalProfit: 321.5,
            maxConcurrentMae: 80,
            maxDrawdown: 40,
            pnlToRisk: 4.02,
            totalDeals: 420,
            profitableDeals: 410,
            losingDeals: 10,
            openDeals: 0,
            activeMae: 0,
            netPerDay: 0.4,
            maxConcurrentPositions: 2,
            backtests: 2,
          },
          folds: [],
          score: 4.02,
          error: null,
        },
        createdAt: 1,
      },
    ],
    backtestCache: {},
    inFlightJobs: [],
    paramStats: {},
    generation: 1,
    backtestsSubmitted: 14,
    status: 'stopped',
    startedAt: 1,
    updatedAt: 2,
    logs: [{ timestamp: 2, level: 'warning', message: 'Запуск остановлен.' }],
  };
};

const candidateMetrics = (pnlToRisk: number): PortfolioMetrics => ({
  totalProfit: pnlToRisk * 100,
  maxConcurrentMae: 80,
  maxDrawdown: 40,
  pnlToRisk,
  totalDeals: 420,
  profitableDeals: 410,
  losingDeals: 10,
  openDeals: 0,
  activeMae: 0,
  netPerDay: 0.4,
  maxConcurrentPositions: 2,
  backtests: 2,
});

/** Сохранённый запуск с двумя кандидатами, посчитанными на полной корзине. */
const createSavedStateWithCandidates = (): PortfolioOptimizerState => {
  const state = createSavedState();
  const base = state.candidates[0];
  const candidate = (index: number, pnlToRisk: number, ids: [number, number]): Candidate => ({
    ...base,
    id: `posaved-g1-${index}`,
    hash: `pg${index}`,
    parentId: base.id,
    generation: 1,
    full: base.full
      ? {
          ...base.full,
          backtestIds: { ADA: ids[0], XRP: ids[1] },
          metrics: candidateMetrics(pnlToRisk),
          score: pnlToRisk,
        }
      : null,
  });
  return {
    ...state,
    bestCandidateId: 'posaved-g1-1',
    candidates: [...state.candidates, candidate(1, 5, [31, 32]), candidate(2, 4.5, [41, 42])],
  };
};

// ═══════════════════════════════════════════════════════════════════
// РЕНДЕР
// ═══════════════════════════════════════════════════════════════════

let container: HTMLDivElement;
let root: Root;

const flush = async () => {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
};

const render = async (extensionReady: boolean) => {
  await act(async () => {
    root.render(
      <MemoryRouter>
        <BacktestGroupsProvider>
          <PortfolioOptimizerPage extensionReady={extensionReady} />
        </BacktestGroupsProvider>
      </MemoryRouter>,
    );
  });
  await flush();
};

const findButton = (label: string): HTMLButtonElement => {
  const button = Array.from(container.querySelectorAll('button')).find(
    (item) => item.textContent?.replace(/\s+/g, ' ').trim() === label,
  );
  if (!button) {
    throw new Error(`Кнопка «${label}» не найдена`);
  }
  return button;
};

const findByLabel = (label: string): HTMLButtonElement => {
  const button = document.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);
  if (!button) {
    throw new Error(`Кнопка «${label}» не найдена`);
  }
  return button;
};

/** Подтверждение в последнем открытом Popconfirm. */
const confirmPopover = async () => {
  const popovers = Array.from(document.querySelectorAll('.ant-popover:not(.ant-popover-hidden)'));
  const ok = popovers.at(-1)?.querySelector<HTMLButtonElement>('.ant-btn-primary');
  if (!ok) {
    throw new Error('Нет открытого подтверждения');
  }
  await act(async () => {
    ok.click();
  });
  await flush();
};

const setTextAreaValue = async (element: HTMLTextAreaElement, value: string) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
  await act(async () => {
    setter?.call(element, value);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
};

beforeEach(() => {
  window.localStorage.clear();
  fetchBotsMock.mockReset();
  createBotMock.mockReset();
  fetchBotStrategyMock.mockReset();
  fetchBotsMock.mockResolvedValue({
    content: [{ id: 77, name: 'LONG 22', algorithm: 'LONG', exchange: 'BYBIT_FUTURES', symbols: ['ADA/USDT'] }],
    totalElements: 1,
    totalPages: 1,
    pageNumber: 0,
    size: 100,
  });
  fetchBotStrategyMock.mockResolvedValue(createStrategy());
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  container.remove();
  portfolioRunSession.clear();
});

// ═══════════════════════════════════════════════════════════════════
// ТЕСТЫ
// ═══════════════════════════════════════════════════════════════════

describe('PortfolioOptimizerPage', () => {
  it('рендерится вне расширения: предупреждение, все разделы, старт недоступен', async () => {
    await render(false);
    const text = container.textContent ?? '';
    expect(text).toContain('Портфельный оптимизатор');
    expect(text).toContain('Расширение недоступно');
    for (const title of [
      '1. Базовый бот',
      '2. Корзина монет',
      '3. Период и комиссии',
      '4. Цель',
      '5. Поиск и лимиты',
      '6. Параметры',
      '7. Запуск',
      '8. Кандидаты',
      '9. Журнал',
    ]) {
      expect(text).toContain(title);
    }
    expect(findButton('Старт').disabled).toBe(true);
    expect(fetchBotsMock).not.toHaveBeenCalled();
    expect(fetchBotStrategyMock).not.toHaveBeenCalled();
  });

  it('загружает ботов и стратегию последнего бота, заполняет корзину и параметры', async () => {
    window.localStorage.setItem(PORTFOLIO_FORM_STORAGE_KEY, JSON.stringify({ version: 1, botId: 77 }));
    await render(true);

    expect(fetchBotsMock).toHaveBeenCalledWith({ page: 0, size: 100 });
    expect(fetchBotStrategyMock).toHaveBeenCalledWith(77);
    const text = container.textContent ?? '';
    expect(text).toContain('Загружен «LONG 22» (#77)');
    expect(text).toContain('котировка USDT');
    // Параметры бота в таблице.
    expect(text).toContain('RSI 1 час < 40');
    expect(text).toContain('CCI 1 час < -100');
    // Корзина заполнена монетами бота и сохранена в форме.
    const textarea = container.querySelector<HTMLTextAreaElement>('textarea[aria-label="Монеты корзины"]');
    expect(textarea?.value).toBe('ADA, XRP');
    const stored = JSON.parse(window.localStorage.getItem(PORTFOLIO_FORM_STORAGE_KEY) ?? '{}');
    expect(stored.basketText).toBe('ADA, XRP');
    expect(stored.botId).toBe(77);
    // Всё готово к запуску.
    expect(findButton('Старт').disabled).toBe(false);
    expect(text).not.toContain('Перед стартом');
  });

  it('«Старт» передаёт сеансу конфигурацию из формы и стратегию бота', async () => {
    const start = vi.spyOn(portfolioRunSession, 'start').mockReturnValue(Promise.resolve());
    window.localStorage.setItem(
      PORTFOLIO_FORM_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        botId: 77,
        basketText: 'ADA, XRP, SOL',
        screenSize: 2,
        periodFrom: '2022-01-01',
        periodTo: '2024-01-01',
        block: 5,
      }),
    );
    await render(true);

    await act(async () => {
      findButton('Старт').click();
    });
    await flush();
    expect(start).toHaveBeenCalledTimes(1);
    const input = start.mock.calls[0][0];
    expect(input.quoteCurrency).toBe('USDT');
    expect(input.baseStrategy).toEqual(createStrategy());
    expect(input.config).toMatchObject({
      botId: 77,
      botName: 'LONG 22',
      basket: ['ADA', 'XRP', 'SOL'],
      block: 5,
      baseGroupId: null,
      run: { periodFrom: '2022-01-01T00:00:00.000Z', periodTo: '2024-01-01T23:59:59.999Z' },
    });
    expect(input.config.screenBasket).toHaveLength(2);
    expect(input.config.params).toEqual(extractParams(createStrategy()));
  });

  it('сохраняет изменения формы в localStorage', async () => {
    await render(true);
    const textarea = container.querySelector<HTMLTextAreaElement>('textarea[aria-label="Монеты корзины"]');
    if (!textarea) {
      throw new Error('нет поля корзины');
    }
    await setTextAreaValue(textarea, 'sol, doge link');
    const stored = JSON.parse(window.localStorage.getItem(PORTFOLIO_FORM_STORAGE_KEY) ?? '{}');
    expect(stored.basketText).toBe('sol, doge link');
    expect(container.textContent).toContain('Монет: 3.');
    expect(container.textContent).toContain('Отборочная корзина (3)');
    // Бот не выбран — запуск невозможен, причина показана.
    expect(container.textContent).toContain('Выберите базового бота.');
  });

  it('показывает сохранённый запуск: баннер продолжения и результаты', async () => {
    window.localStorage.setItem(PORTFOLIO_OPTIMIZER_STATE_KEY, JSON.stringify(createSavedState()));
    await render(true);

    const text = container.textContent ?? '';
    expect(text).toContain('Найден незавершённый запуск портфельного оптимизатора');
    expect(text).toContain('поколение 1 из 3');
    expect(findButton('Продолжить').disabled).toBe(false);
    // Результаты сохранённого запуска видны без продолжения.
    expect(text).toContain('Лучший результат — база');
    expect(text).toContain('321.50');
    expect(text).toContain('Запуск остановлен.');
    expect(text).toContain('Остановлен');
  });

  it('группа бэктестов кандидата открывается с блокировкой по ботам, с которой его считал оптимизатор', async () => {
    window.localStorage.setItem(PORTFOLIO_OPTIMIZER_STATE_KEY, JSON.stringify(createSavedStateWithCandidates()));
    await render(true);

    await act(async () => {
      findButton('Группа бэктестов').click();
    });
    const group = readBacktestGroups().find((item) => item.name.endsWith('g1-1'));
    expect(group?.backtestIds).toEqual([31, 32]);
    // Блокировка запуска 60, но в группе только 2 бэктеста — как страница группы, ограничиваем их числом.
    expect(readAggregationConfig(group?.id ?? '')).toEqual({ maxConcurrentPositions: 2 });
    expect(document.body.textContent).toContain('блокировка по ботам 2 — как в оптимизаторе');
  });

  it('«Продолжить» читает сохранённый запуск в момент нажатия и применяет потоки и интервал из формы', async () => {
    const resume = vi.spyOn(portfolioRunSession, 'resume').mockReturnValue(Promise.resolve());
    window.localStorage.setItem(PORTFOLIO_OPTIMIZER_STATE_KEY, JSON.stringify(createSavedState()));
    window.localStorage.setItem(
      PORTFOLIO_FORM_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        scheduler: { maxInFlight: 2, submitDelaySeconds: 45, pollIntervalSeconds: 10, backtestTimeoutMinutes: 60 },
      }),
    );
    await render(true);
    // Пока страница открыта, состояние дописали (например, бэктест, созданный после остановки).
    window.localStorage.setItem(
      PORTFOLIO_OPTIMIZER_STATE_KEY,
      JSON.stringify({ ...createSavedState(), backtestsSubmitted: 15 }),
    );

    await act(async () => {
      findButton('Продолжить').click();
    });
    await flush();
    expect(resume).toHaveBeenCalledTimes(1);
    const state = resume.mock.calls[0][0];
    expect(state.backtestsSubmitted).toBe(15);
    expect(state.config.scheduler).toEqual({
      maxInFlight: 2,
      submitDelaySeconds: 45,
      pollIntervalSeconds: 10,
      backtestTimeoutMinutes: 60,
    });
    expect(state.config.block).toBe(60);
  });

  it('запуск, который держит блокировку другой вкладки, нельзя продолжить или заменить здесь', async () => {
    const detect = vi.spyOn(portfolioRunSession, 'detectOtherTabRun').mockResolvedValue('locked');
    const resume = vi.spyOn(portfolioRunSession, 'resume').mockReturnValue(Promise.resolve());
    const start = vi.spyOn(portfolioRunSession, 'start').mockReturnValue(Promise.resolve());
    window.localStorage.setItem(
      PORTFOLIO_OPTIMIZER_STATE_KEY,
      JSON.stringify({ ...createSavedState(), status: 'paused' }),
    );
    window.localStorage.setItem(
      PORTFOLIO_FORM_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        botId: 77,
        basketText: 'ADA, XRP',
        periodFrom: '2022-01-01',
        periodTo: '2024-01-01',
      }),
    );
    await render(true);

    const text = container.textContent ?? '';
    expect(detect).toHaveBeenCalled();
    expect(text).toContain('статус «выполняется в другой вкладке»');
    expect(text).not.toContain('«прерван»');
    expect(findButton('Продолжить').disabled).toBe(true);

    await act(async () => {
      findButton('Старт').click();
    });
    await flush();
    expect(start).not.toHaveBeenCalled();
    expect(resume).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain('Оптимизатор уже работает в другой вкладке');
  });

  it('отдельно показывает загрузку создания бота для каждого кандидата', async () => {
    const pending: Array<(value: { id: number }) => void> = [];
    createBotMock.mockImplementation(
      () =>
        new Promise<{ id: number }>((resolve) => {
          pending.push(resolve);
        }),
    );
    const saved = createSavedStateWithCandidates();
    // apiKey приходит в полном DTO бота (в типе BotStrategy его нет) — без него бота не создать.
    window.localStorage.setItem(
      PORTFOLIO_OPTIMIZER_STATE_KEY,
      JSON.stringify({ ...saved, baseStrategy: { ...saved.baseStrategy, apiKey: 5 } }),
    );
    await render(true);
    const isLoading = (label: string) => findByLabel(label).classList.contains('ant-btn-loading');

    await act(async () => {
      findByLabel('Создать бота g1-1').click();
    });
    await flush();
    await confirmPopover();
    await act(async () => {
      findByLabel('Создать бота g1-2').click();
    });
    await confirmPopover();
    expect(createBotMock).toHaveBeenCalledTimes(2);
    expect(isLoading('Создать бота g1-1')).toBe(true);
    expect(isLoading('Создать бота g1-2')).toBe(true);

    // Второй бот создан — загрузка первого не пропадает.
    await act(async () => {
      pending[1]({ id: 902 });
    });
    await flush();
    expect(isLoading('Создать бота g1-1')).toBe(true);
    expect(isLoading('Создать бота g1-2')).toBe(false);
    await act(async () => {
      pending[0]({ id: 901 });
    });
    await flush();
    expect(isLoading('Создать бота g1-1')).toBe(false);
  });

  it('предупреждает, если блокировка группы базы отличается от формы, и берёт её по кнопке', async () => {
    writeBacktestGroups([{ id: 'g-base', name: 'База LONG', backtestIds: [1, 2, 3, 4], createdAt: 1, updatedAt: 1 }]);
    writeAggregationConfig('g-base', { maxConcurrentPositions: 3 });
    window.localStorage.setItem(PORTFOLIO_FORM_STORAGE_KEY, JSON.stringify({ version: 1, baseGroupId: 'g-base' }));
    await render(true);

    expect(container.textContent).toContain('В группе «База LONG» блокировка по ботам 3, в форме — 60');
    await act(async () => {
      findButton('Взять из группы').click();
    });
    expect(container.textContent).not.toContain('В группе «База LONG» блокировка по ботам');
    expect(JSON.parse(window.localStorage.getItem(PORTFOLIO_FORM_STORAGE_KEY) ?? '{}').block).toBe(3);
  });

  it('не даёт начать запуск, если лимита бэктестов не хватит на базу', async () => {
    window.localStorage.setItem(
      PORTFOLIO_FORM_STORAGE_KEY,
      JSON.stringify({ version: 1, botId: 77, basketText: 'ADA, XRP, SOL', search: { maxBacktests: 2 } }),
    );
    await render(true);
    expect(container.textContent).toContain('Лимит бэктестов (2) меньше, чем нужно для оценки базы (3)');
    expect(findButton('Старт').disabled).toBe(true);
  });
});
