/**
 * Страница портфельного оптимизатора («уточнение базового бота»).
 *
 * Берёт реальный DTO бота Veles, предлагает к поиску его числовые параметры (пороги условий,
 * сетку, тейк, стоп, подтяжку) и прогоняет кандидатов бэктестами Veles: сначала на отборочной
 * корзине, лучших — на полной. Кандидаты сравниваются по метрикам группы бэктестов (P&L к риску
 * по МПУ). Запуск выполняет PortfolioOptimizer через общий сеанс (runSession), поэтому он
 * продолжается при переходе на другие страницы расширения и восстанавливается после перезагрузки.
 */

import {
  ImportOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  RetweetOutlined,
  StopOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  Flex,
  Input,
  InputNumber,
  Modal,
  message,
  Popconfirm,
  Row,
  Select,
  Statistic,
  Switch,
  Tag,
  Typography,
} from 'antd';
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link } from 'react-router-dom';
import { type BotStrategy, fetchBotStrategy, resolveQuoteCurrency } from '../api/backtestRunner';
import { createBot, fetchBots } from '../api/bots';
import BestCandidateCard from '../components/portfolioOptimizer/BestCandidateCard';
import CandidatesTable from '../components/portfolioOptimizer/CandidatesTable';
import FormField from '../components/portfolioOptimizer/FormField';
import {
  applyParamOverrides,
  basketFromSymbols,
  buildPortfolioConfig,
  buildRunSettings,
  collectParamOverrides,
  DEFAULT_QUOTE_CURRENCY,
  FORM_LIMITS,
  type PortfolioOptimizerFormState,
  parseBasket,
  readPortfolioFormState,
  resolveScreenBasket,
  resolveScreenSize,
  sameParamOverrides,
  validateRunDraft,
  writePortfolioFormState,
} from '../components/portfolioOptimizer/formState';
import ParamsTable from '../components/portfolioOptimizer/ParamsTable';
import PortfolioLogViewer from '../components/portfolioOptimizer/PortfolioLogViewer';
import RunProgressPanel from '../components/portfolioOptimizer/RunProgressPanel';
import type { OtherTabRun } from '../components/portfolioOptimizer/runLock';
import { portfolioRunSession } from '../components/portfolioOptimizer/runSession';
import {
  buildBestSummary,
  buildCandidateGroupName,
  buildCandidateRows,
  buildOptimizedBotName,
  type CandidateActionTarget,
  formatDuration,
  withGroupMissing,
  withSchedulerConfig,
} from '../components/portfolioOptimizer/runView';
import SavedRunBanner from '../components/portfolioOptimizer/SavedRunBanner';
import PageHeader from '../components/ui/PageHeader';
import { useBacktestGroups } from '../context/BacktestGroupsContext';
import { buildBotCreatePayloadFromStrategy, extractParams, patchStrategy, roundTo } from '../lib/botParams';
import { DEFAULT_AGGREGATION_CONFIG } from '../services/backtestAggregations';
import {
  clearSavedPortfolioState,
  estimateBacktests,
  type GroupBacktestsLoadResult,
  loadGroupBacktestsFromStore,
  loadSavedPortfolioState,
  matchGroupBacktestsToBasket,
} from '../services/portfolioOptimizer';
import { readAggregationConfig, writeAggregationConfig } from '../storage/backtestAggregationConfigStore';
import type { TradingBot } from '../types/bots';
import type {
  Candidate,
  ObjectiveConfig,
  ObjectiveKind,
  ParamSpec,
  PortfolioOptimizerState,
  SchedulerConfig,
  SearchConfig,
} from '../types/portfolioOptimizer';

const { Text } = Typography;
const { TextArea } = Input;

interface PortfolioOptimizerPageProps {
  extensionReady: boolean;
}

/** Стратегия выбранного бота, загруженная из Veles. */
interface LoadedBot {
  botId: number;
  name: string;
  strategy: BotStrategy;
  quoteCurrency: string;
  /** Параметры с границами по умолчанию (для «Сбросить»). */
  defaults: ParamSpec[];
}

interface LoadedGroup {
  groupId: string;
  result: GroupBacktestsLoadResult;
}

const BOTS_PAGE_SIZE = 100;

/** Как часто проверять, не выполняется ли сохранённый запуск в другой вкладке. */
const OTHER_TAB_CHECK_INTERVAL_MS = 10_000;

const OTHER_TAB_BLOCKED_TITLE = 'Оптимизатор уже работает в другой вкладке';
const OTHER_TAB_BLOCKED_CONTENT =
  'Запуск выполняется в другой вкладке расширения. Управляйте им там: остановите его или закройте ту вкладку, ' +
  'прежде чем продолжать или начинать запуск здесь, — иначе два запуска будут создавать бэктесты дважды и ' +
  'перезаписывать сохранённое состояние друг друга.';

const OBJECTIVE_OPTIONS: Array<{ value: ObjectiveKind; label: string }> = [
  { value: 'robust', label: 'Устойчивый P&L к риску (весь период + худший под-период)' },
  { value: 'pnlToRisk', label: 'P&L к риску как в группе' },
  { value: 'profitCapped', label: 'Максимум P&L при МПУ ≤ …' },
];

const OBJECTIVE_HINTS: Readonly<Record<ObjectiveKind, string>> = {
  robust:
    'Score = ½ · P&L/риск за весь период + ½ · худший P&L/риск среди под-периодов: кандидат должен быть хорош ' +
    'на всём периоде, а не только на одном удачном отрезке.',
  pnlToRisk: 'Score = P&L / max(МПУ, просадка) — ровно как «P&L к риску» на странице группы бэктестов.',
  profitCapped:
    'Score = P&L; если МПУ выше ограничения, P&L штрафуется множителем (ограничение / МПУ)². Без ограничения — ' +
    'просто максимум P&L.',
};

const describeError = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const toBotId = (value: TradingBot['id']): number | null => {
  const id = typeof value === 'number' ? value : Number(value);
  return Number.isInteger(id) && id > 0 ? id : null;
};

/** Значение InputNumber: null (поле очищено) не меняет настройку. */
const whenNumber =
  (apply: (value: number) => void) =>
  (value: number | null): void => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      apply(value);
    }
  };

const PortfolioOptimizerPage = ({ extensionReady }: PortfolioOptimizerPageProps) => {
  const [messageApi, messageContextHolder] = message.useMessage();
  const [modal, modalContextHolder] = Modal.useModal();
  const { groups, createGroup } = useBacktestGroups();
  const snapshot = useSyncExternalStore(portfolioRunSession.subscribe, portfolioRunSession.getSnapshot);
  const now = Date.now();

  // ─── Форма ───────────────────────────────────────────────────────

  const [form, setForm] = useState<PortfolioOptimizerFormState>(() => readPortfolioFormState());
  const formRef = useRef(form);

  useEffect(() => {
    formRef.current = form;
    writePortfolioFormState(form);
  }, [form]);

  const updateForm = useCallback((patch: Partial<PortfolioOptimizerFormState>) => {
    setForm((prev) => ({ ...prev, ...patch }));
  }, []);
  const updateSearch = useCallback((patch: Partial<SearchConfig>) => {
    setForm((prev) => ({ ...prev, search: { ...prev.search, ...patch } }));
  }, []);
  const updateObjective = useCallback((patch: Partial<ObjectiveConfig>) => {
    setForm((prev) => ({ ...prev, objective: { ...prev.objective, ...patch } }));
  }, []);
  const updateScheduler = useCallback((patch: Partial<SchedulerConfig>) => {
    setForm((prev) => ({ ...prev, scheduler: { ...prev.scheduler, ...patch } }));
  }, []);

  // ─── Боты ────────────────────────────────────────────────────────

  const [bots, setBots] = useState<TradingBot[]>([]);
  const [botsLoading, setBotsLoading] = useState(false);
  const [botsError, setBotsError] = useState<string | null>(null);

  const loadBots = useCallback(async () => {
    setBotsLoading(true);
    setBotsError(null);
    try {
      const response = await fetchBots({ page: 0, size: BOTS_PAGE_SIZE });
      setBots(response.content);
    } catch (error) {
      setBotsError(describeError(error));
    } finally {
      setBotsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (extensionReady) {
      void loadBots();
    }
  }, [extensionReady, loadBots]);

  // ─── Стратегия базового бота ─────────────────────────────────────

  const [loaded, setLoaded] = useState<LoadedBot | null>(null);
  const [params, setParams] = useState<ParamSpec[]>([]);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [strategyError, setStrategyError] = useState<string | null>(null);
  const strategyRequestRef = useRef(0);

  const loadStrategy = useCallback(async (botId: number) => {
    strategyRequestRef.current += 1;
    const requestId = strategyRequestRef.current;
    setStrategyLoading(true);
    setStrategyError(null);
    try {
      const strategy = await fetchBotStrategy(botId);
      if (strategyRequestRef.current !== requestId) {
        return;
      }
      const defaults = extractParams(strategy);
      const current = formRef.current;
      setLoaded({
        botId,
        name: strategy.name?.trim() || `Бот #${botId}`,
        strategy,
        quoteCurrency: resolveQuoteCurrency(strategy) ?? DEFAULT_QUOTE_CURRENCY,
        defaults,
      });
      setParams(applyParamOverrides(defaults, current.paramOverrides, botId));
      if (parseBasket(current.basketText).length === 0) {
        const basket = basketFromSymbols(strategy.symbols ?? [strategy.symbol]);
        if (basket.length > 0) {
          setForm((prev) => ({ ...prev, basketText: basket.join(', ') }));
        }
      }
    } catch (error) {
      if (strategyRequestRef.current === requestId) {
        setStrategyError(describeError(error));
      }
    } finally {
      if (strategyRequestRef.current === requestId) {
        setStrategyLoading(false);
      }
    }
  }, []);

  // Стратегия последнего выбранного бота подгружается при открытии страницы.
  const autoLoadedRef = useRef(false);
  useEffect(() => {
    if (!extensionReady || autoLoadedRef.current) {
      return;
    }
    autoLoadedRef.current = true;
    const botId = formRef.current.botId;
    if (botId !== null) {
      void loadStrategy(botId);
    }
  }, [extensionReady, loadStrategy]);

  // Правки границ параметров запоминаются в форме (для того же бота).
  useEffect(() => {
    if (!loaded) {
      return;
    }
    const overrides = collectParamOverrides(loaded.botId, params, loaded.defaults);
    setForm((prev) =>
      sameParamOverrides(prev.paramOverrides, overrides) ? prev : { ...prev, paramOverrides: overrides },
    );
  }, [loaded, params]);

  const botOptions = useMemo(() => {
    const options = bots.flatMap((bot) => {
      const id = toBotId(bot.id);
      return id === null ? [] : [{ value: id, label: `${bot.name} (${bot.algorithm}, ${bot.exchange})` }];
    });
    if (loaded && !options.some((option) => option.value === loaded.botId)) {
      options.unshift({ value: loaded.botId, label: loaded.name });
    }
    return options;
  }, [bots, loaded]);

  const selectedBotSymbols = useMemo((): string[] => {
    if (loaded && loaded.botId === form.botId) {
      return basketFromSymbols(loaded.strategy.symbols ?? [loaded.strategy.symbol]);
    }
    const bot = bots.find((item) => toBotId(item.id) === form.botId);
    return bot ? basketFromSymbols(bot.symbols) : [];
  }, [bots, form.botId, loaded]);

  // ─── Корзина, группа базы, оценка ────────────────────────────────

  const basket = useMemo(() => parseBasket(form.basketText), [form.basketText]);
  const screenSize = resolveScreenSize(basket.length, form.screenSize);
  const screenBasket = useMemo(
    () => resolveScreenBasket({ screenSize: form.screenSize, screenSeed: form.screenSeed }, basket),
    [basket, form.screenSeed, form.screenSize],
  );
  const baseGroup = useMemo(
    () => groups.find((group) => group.id === form.baseGroupId) ?? null,
    [form.baseGroupId, groups],
  );
  /** Форма с проверенной группой базы (удалённая группа не используется). */
  const effectiveForm = useMemo(() => ({ ...form, baseGroupId: baseGroup?.id ?? null }), [baseGroup, form]);
  const runSettings = useMemo(() => buildRunSettings(form), [form]);
  const quoteCurrency = loaded?.quoteCurrency ?? DEFAULT_QUOTE_CURRENCY;

  const [loadedGroup, setLoadedGroup] = useState<LoadedGroup | null>(null);
  const [groupLoading, setGroupLoading] = useState(false);
  const [groupError, setGroupError] = useState<string | null>(null);

  const handleCheckGroup = useCallback(async () => {
    if (!baseGroup) {
      return;
    }
    setGroupLoading(true);
    setGroupError(null);
    try {
      const result = await loadGroupBacktestsFromStore(baseGroup.id);
      setLoadedGroup({ groupId: baseGroup.id, result });
    } catch (error) {
      setLoadedGroup(null);
      setGroupError(describeError(error));
    } finally {
      setGroupLoading(false);
    }
  }, [baseGroup]);

  const groupMatch = useMemo(() => {
    if (!(loadedGroup && baseGroup && loadedGroup.groupId === baseGroup.id && runSettings.periodFrom)) {
      return null;
    }
    return matchGroupBacktestsToBasket({
      backtests: loadedGroup.result.backtests,
      basket,
      quoteCurrency,
      run: runSettings,
      algorithm: loaded?.strategy.algorithm ?? null,
      baseStrategy: loaded?.strategy ?? null,
      params: loaded ? params : undefined,
    });
  }, [baseGroup, basket, loaded, loadedGroup, params, quoteCurrency, runSettings]);

  /** Блокировка по ботам, с которой страница группы базы показывает её метрики (как BacktestGroupDetailsPage). */
  const baseGroupBlock = useMemo(() => {
    if (!baseGroup) {
      return null;
    }
    const stored = readAggregationConfig(baseGroup.id) ?? DEFAULT_AGGREGATION_CONFIG;
    return Math.min(Math.max(1, stored.maxConcurrentPositions), Math.max(1, baseGroup.backtestIds.length));
  }, [baseGroup]);

  const draftConfig = useMemo(
    () => buildPortfolioConfig({ form: effectiveForm, botId: form.botId ?? 0, botName: loaded?.name ?? '', params }),
    [effectiveForm, form.botId, loaded, params],
  );
  const estimate = useMemo(() => {
    const base = estimateBacktests(draftConfig);
    return groupMatch
      ? withGroupMissing(
          base,
          groupMatch.missing.length,
          draftConfig.search.maxBacktests,
          draftConfig.scheduler.submitDelaySeconds,
        )
      : base;
  }, [draftConfig, groupMatch]);

  // ─── Запуск ──────────────────────────────────────────────────────

  const active = snapshot?.active ?? false;
  const runStatus = snapshot?.progress.status ?? 'idle';
  const meta = snapshot?.meta ?? null;
  const [savedState, setSavedState] = useState<PortfolioOptimizerState | null>(() =>
    portfolioRunSession.isActive ? null : loadSavedPortfolioState(),
  );

  // Результаты последнего запуска видны сразу после открытия страницы.
  useEffect(() => {
    if (savedState && !portfolioRunSession.getSnapshot()) {
      portfolioRunSession.show(savedState);
    }
  }, [savedState]);

  // После окончания запуска сохранённое состояние перечитывается (для «Продолжить»/«Удалить»).
  useEffect(() => {
    if (!active) {
      setSavedState(loadSavedPortfolioState());
    }
  }, [active]);

  const problems = useMemo(
    () =>
      validateRunDraft({
        form: effectiveForm,
        loadedBotId: loaded?.botId ?? null,
        params,
        // С группой базы цена базы известна только после «Проверить» (недостающие монеты).
        baseBacktests: baseGroup ? (groupMatch ? groupMatch.missing.length : null) : basket.length,
      }),
    [baseGroup, basket.length, effectiveForm, groupMatch, loaded, params],
  );

  // Выполняется ли сохранённый запуск в другой вкладке (Web Locks или «пульс»): проверяется периодически.
  const [elsewhere, setElsewhere] = useState<OtherTabRun>('none');
  useEffect(() => {
    if (!savedState || active) {
      setElsewhere('none');
      return;
    }
    let cancelled = false;
    const check = (): void => {
      void portfolioRunSession.detectOtherTabRun(savedState, Date.now()).then((value) => {
        if (!cancelled) {
          setElsewhere(value);
        }
      });
    };
    check();
    const timer = setInterval(check, OTHER_TAB_CHECK_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [active, savedState]);

  const launchNewRun = useCallback(() => {
    if (!loaded) {
      return;
    }
    try {
      const config = buildPortfolioConfig({
        form: effectiveForm,
        botId: loaded.botId,
        botName: loaded.name,
        params,
      });
      void portfolioRunSession.start({ config, baseStrategy: loaded.strategy, quoteCurrency: loaded.quoteCurrency });
      setSavedState(null);
    } catch (error) {
      messageApi.error(describeError(error));
    }
  }, [effectiveForm, loaded, messageApi, params]);

  const handleStart = useCallback(async () => {
    if (problems.length > 0) {
      messageApi.error(problems[0]);
      return;
    }
    const stored = loadSavedPortfolioState();
    const otherTab = await portfolioRunSession.detectOtherTabRun(stored, Date.now());
    if (otherTab === 'locked') {
      modal.warning({ title: OTHER_TAB_BLOCKED_TITLE, content: OTHER_TAB_BLOCKED_CONTENT });
      return;
    }
    if (stored) {
      modal.confirm({
        title: 'Начать новый запуск?',
        content:
          (otherTab === 'probable'
            ? 'Похоже, сохранённый запуск выполняется в другой вкладке расширения — новый запуск здесь заменит его ' +
              'состояние, а та вкладка продолжит создавать бэктесты. '
            : '') +
          'Сохранённый запуск будет заменён новым: продолжить его или посмотреть его результаты станет нельзя. ' +
          'Бэктесты и боты в Veles останутся.',
        okText: 'Начать',
        cancelText: 'Отмена',
        onOk: launchNewRun,
      });
      return;
    }
    launchNewRun();
  }, [launchNewRun, messageApi, modal, problems]);

  const resumeSaved = useCallback(
    (state: PortfolioOptimizerState) => {
      try {
        // Потоки, интервал и таймаут, изменённые в форме перед «Продолжить», применяются к продолжению.
        void portfolioRunSession.resume(withSchedulerConfig(state, formRef.current.scheduler));
        setSavedState(null);
      } catch (error) {
        messageApi.error(describeError(error));
      }
    },
    [messageApi],
  );

  const handleResumeSaved = useCallback(async () => {
    // Состояние читается в момент нажатия: его могли дописать после загрузки страницы (например, бэктест,
    // ответ на создание которого пришёл после остановки).
    const state = loadSavedPortfolioState();
    if (!state) {
      setSavedState(null);
      messageApi.warning('Сохранённый запуск не найден — возможно, его удалили в другой вкладке.');
      return;
    }
    const otherTab = await portfolioRunSession.detectOtherTabRun(state, Date.now());
    if (otherTab === 'locked') {
      setElsewhere('locked');
      modal.warning({ title: OTHER_TAB_BLOCKED_TITLE, content: OTHER_TAB_BLOCKED_CONTENT });
      return;
    }
    if (otherTab === 'probable') {
      modal.confirm({
        title: 'Продолжить запуск здесь?',
        content:
          'Похоже, этот запуск выполняется в другой вкладке расширения. Если она открыта, оптимизация пойдёт ' +
          'дважды и потратит лишние бэктесты.',
        okText: 'Продолжить здесь',
        cancelText: 'Отмена',
        onOk: () => resumeSaved(state),
      });
      return;
    }
    resumeSaved(state);
  }, [messageApi, modal, resumeSaved]);

  const handleClearSaved = useCallback(() => {
    clearSavedPortfolioState();
    portfolioRunSession.clear();
    setSavedState(null);
    messageApi.info('Сохранённый запуск удалён.');
  }, [messageApi]);

  const handleSchedulerChange = useCallback(
    (patch: Partial<SchedulerConfig>) => {
      updateScheduler(patch);
      if (active) {
        portfolioRunSession.updateSchedulerConfig(patch);
      }
    },
    [active, updateScheduler],
  );

  const schedulerValues = active && meta ? meta.config.scheduler : form.scheduler;

  // ─── Результаты ──────────────────────────────────────────────────

  const candidates = snapshot?.candidates;
  const bestCandidateId = snapshot?.bestCandidateId ?? null;
  const currentGeneration = snapshot?.progress.generation ?? 0;

  const rows = useMemo(
    () =>
      meta && candidates
        ? buildCandidateRows({
            runId: meta.runId,
            baseCandidateId: meta.baseCandidateId,
            params: meta.config.params,
            candidates,
            bestCandidateId,
            currentGeneration,
            active,
          })
        : [],
    [active, bestCandidateId, candidates, currentGeneration, meta],
  );

  const bestSummary = useMemo(
    () =>
      meta && candidates
        ? buildBestSummary({
            runId: meta.runId,
            baseCandidateId: meta.baseCandidateId,
            params: meta.config.params,
            candidates,
            bestCandidateId,
          })
        : null,
    [bestCandidateId, candidates, meta],
  );

  const runExpectedBacktests = useMemo(() => (meta ? estimateBacktests(meta.config).totalMax : 0), [meta]);

  const strategyFor = useCallback(
    (candidate: Candidate): BotStrategy | null =>
      meta ? patchStrategy(meta.baseStrategy, meta.config.params, candidate.values) : null,
    [meta],
  );

  const handleCreateGroup = useCallback(
    (target: CandidateActionTarget) => {
      if (!(meta && target.backtests)) {
        return;
      }
      const group = createGroup(buildCandidateGroupName(meta.config.botName, target), target.backtests.ids);
      if (!group) {
        messageApi.error('Не удалось создать группу бэктестов.');
        return;
      }
      // Группа открывается с той же блокировкой по ботам, с которой оптимизатор считал кандидата.
      const block = Math.max(1, Math.min(meta.config.block, group.backtestIds.length));
      writeAggregationConfig(group.id, { maxConcurrentPositions: block });
      messageApi.success({
        content: (
          <span>
            Группа «{group.name}» создана ({group.backtestIds.length} бэктестов, блокировка по ботам {block} — как в
            оптимизаторе). <Link to={`/backtest-groups/${group.id}`}>Открыть группу</Link>
          </span>
        ),
        duration: 8,
      });
    },
    [createGroup, messageApi, meta],
  );

  const handleCopyJson = useCallback(
    async (target: CandidateActionTarget) => {
      const strategy = strategyFor(target.candidate);
      if (!strategy) {
        return;
      }
      if (typeof navigator === 'undefined' || !navigator.clipboard?.writeText) {
        messageApi.error('Буфер обмена недоступен в этом окне.');
        return;
      }
      try {
        await navigator.clipboard.writeText(JSON.stringify(strategy, null, 2));
        messageApi.success(`JSON стратегии ${target.label} скопирован в буфер обмена.`);
      } catch (error) {
        messageApi.error(`Не удалось скопировать JSON: ${describeError(error)}`);
      }
    },
    [messageApi, strategyFor],
  );

  // Боты, которые создаются прямо сейчас: запросы идут через общую очередь и могут занять много секунд,
  // повторное нажатие для того же кандидата игнорируется (иначе в Veles появились бы два одинаковых бота).
  const creatingBotsRef = useRef(new Set<string>());
  const [creatingBots, setCreatingBots] = useState<ReadonlySet<string>>(() => new Set());
  const isCreatingBot = useCallback((candidateId: string) => creatingBots.has(candidateId), [creatingBots]);

  const handleCreateBot = useCallback(
    async (target: CandidateActionTarget) => {
      const strategy = strategyFor(target.candidate);
      const candidateId = target.candidate.id;
      if (!(meta && strategy) || creatingBotsRef.current.has(candidateId)) {
        return;
      }
      const name = buildOptimizedBotName(meta.config.botName, target.label);
      creatingBotsRef.current.add(candidateId);
      setCreatingBots(new Set(creatingBotsRef.current));
      try {
        const response = await createBot(buildBotCreatePayloadFromStrategy(strategy, name));
        messageApi.success(`Бот «${name}» создан (ID ${response.id}).`);
      } catch (error) {
        messageApi.error(`Не удалось создать бота: ${describeError(error)}`);
      } finally {
        creatingBotsRef.current.delete(candidateId);
        setCreatingBots(new Set(creatingBotsRef.current));
      }
    },
    [messageApi, meta, strategyFor],
  );

  // ─── Рендер ──────────────────────────────────────────────────────

  const formLocked = active;
  const resultsBotName = meta?.config.botName ?? loaded?.name ?? '';
  const resultsQuote = meta?.quoteCurrency ?? quoteCurrency;
  const stopping = active && runStatus !== 'running' && runStatus !== 'paused';

  return (
    <section className="page">
      {messageContextHolder}
      {modalContextHolder}
      <PageHeader
        title="Портфельный оптимизатор"
        description={
          'Уточнение параметров существующего бота: кандидаты проверяются бэктестами Veles на корзине монет и ' +
          'сравниваются по метрикам группы бэктестов (P&L к риску по МПУ).'
        }
      />

      {extensionReady ? null : (
        <Alert
          type="warning"
          showIcon
          message="Расширение недоступно"
          description={
            'Откройте страницу из меню расширения Veles Tools: без него нельзя загрузить ботов и запускать ' +
            'бэктесты. Результаты сохранённого запуска доступны для просмотра.'
          }
        />
      )}

      {savedState && !active ? (
        <SavedRunBanner
          state={savedState}
          elsewhere={elsewhere}
          resumeDisabled={!extensionReady}
          onResume={() => void handleResumeSaved()}
          onClear={handleClearSaved}
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Flex vertical gap={16}>
            <Card title="1. Базовый бот" size="small">
              <Flex vertical gap={12}>
                <Flex gap={8}>
                  <Select<number>
                    style={{ flex: 1, minWidth: 0 }}
                    showSearch
                    optionFilterProp="label"
                    placeholder="Выберите бота"
                    loading={botsLoading}
                    value={form.botId ?? undefined}
                    options={botOptions}
                    disabled={formLocked || !extensionReady}
                    onChange={(value) => updateForm({ botId: value })}
                    aria-label="Базовый бот"
                  />
                  <Button
                    icon={<ReloadOutlined />}
                    loading={strategyLoading}
                    disabled={form.botId === null || formLocked || !extensionReady}
                    onClick={() => {
                      if (form.botId !== null) {
                        void loadStrategy(form.botId);
                      }
                    }}
                  >
                    Загрузить
                  </Button>
                </Flex>
                {botsError ? (
                  <Alert
                    type="error"
                    showIcon
                    message="Не удалось загрузить список ботов"
                    description={botsError}
                    action={
                      <Button size="small" onClick={() => void loadBots()}>
                        Повторить
                      </Button>
                    }
                  />
                ) : null}
                {strategyError ? (
                  <Alert
                    type="error"
                    showIcon
                    message="Не удалось загрузить стратегию бота"
                    description={strategyError}
                  />
                ) : null}
                {loaded ? (
                  <Text type="secondary">
                    Загружен «{loaded.name}» (#{loaded.botId}) · {loaded.strategy.algorithm ?? '—'} ·{' '}
                    {loaded.strategy.exchange ?? loaded.strategy.pair?.exchange ?? '—'} · котировка{' '}
                    {loaded.quoteCurrency} · параметров {loaded.defaults.length}
                  </Text>
                ) : (
                  <Text type="secondary">
                    Выберите бота и нажмите «Загрузить»: из его настроек будут взяты параметры для поиска.
                  </Text>
                )}
                {loaded && form.botId !== null && form.botId !== loaded.botId ? (
                  <Alert
                    type="warning"
                    showIcon
                    message="Выбран другой бот — нажмите «Загрузить», чтобы взять его стратегию."
                  />
                ) : null}
              </Flex>
            </Card>

            <Card title="2. Корзина монет" size="small">
              <Flex vertical gap={12}>
                <FormField
                  label="Монеты корзины"
                  extra={`Монет: ${basket.length}. Разделители — запятая, пробел или перевод строки; котировку («/USDT») можно не указывать.`}
                >
                  <TextArea
                    rows={3}
                    value={form.basketText}
                    placeholder="ADA, XRP, SOL, DOGE, LINK…"
                    disabled={formLocked}
                    onChange={(event) => updateForm({ basketText: event.target.value })}
                    aria-label="Монеты корзины"
                  />
                </FormField>
                <Flex gap={8} wrap>
                  <Button
                    icon={<ImportOutlined />}
                    disabled={formLocked || selectedBotSymbols.length === 0}
                    onClick={() => updateForm({ basketText: selectedBotSymbols.join(', ') })}
                  >
                    Из бота
                  </Button>
                </Flex>
                <Row gutter={[12, 12]} align="bottom">
                  <Col xs={24} sm={14}>
                    <FormField
                      label="Размер отборочной корзины"
                      hint="Каждый кандидат сначала проверяется на этих монетах; лучшие по отбору — на всей корзине."
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={1}
                        max={Math.max(1, basket.length)}
                        value={basket.length > 0 ? screenSize : null}
                        disabled={formLocked || basket.length === 0}
                        onChange={(value) => updateForm({ screenSize: value ?? null })}
                        aria-label="Размер отборочной корзины"
                      />
                    </FormField>
                  </Col>
                  <Col xs={24} sm={10}>
                    <Button
                      block
                      icon={<RetweetOutlined />}
                      disabled={formLocked || screenSize >= basket.length}
                      onClick={() => updateForm({ screenSeed: form.screenSeed + 1 })}
                    >
                      Перемешать
                    </Button>
                  </Col>
                </Row>
                <FormField label={`Отборочная корзина (${screenBasket.length})`}>
                  {screenBasket.length > 0 ? (
                    <Flex wrap gap={4}>
                      {screenBasket.map((ticker) => (
                        <Tag key={ticker} style={{ marginInlineEnd: 0 }}>
                          {ticker}
                        </Tag>
                      ))}
                    </Flex>
                  ) : (
                    <Text type="secondary">Добавьте монеты в корзину.</Text>
                  )}
                </FormField>
                <FormField
                  label="Готовые результаты базы из группы бэктестов"
                  hint="Если базовый бот уже посчитан на монетах корзины за тот же период, его бэктесты возьмутся из группы — это экономит бэктесты базы. Недостающие монеты будут посчитаны."
                >
                  <Flex gap={8}>
                    <Select<string>
                      style={{ flex: 1, minWidth: 0 }}
                      allowClear
                      showSearch
                      optionFilterProp="label"
                      placeholder="Не использовать — база будет посчитана бэктестами"
                      value={baseGroup?.id}
                      options={groups.map((group) => ({
                        value: group.id,
                        label: `${group.name} (${group.backtestIds.length})`,
                      }))}
                      disabled={formLocked}
                      onChange={(value) => {
                        updateForm({ baseGroupId: value ?? null });
                        setLoadedGroup(null);
                        setGroupError(null);
                      }}
                      aria-label="Группа бэктестов базы"
                    />
                    <Button
                      loading={groupLoading}
                      disabled={!(baseGroup && extensionReady)}
                      onClick={() => void handleCheckGroup()}
                    >
                      Проверить
                    </Button>
                  </Flex>
                </FormField>
                {groupError ? (
                  <Alert type="error" showIcon message="Не удалось проверить группу" description={groupError} />
                ) : null}
                {baseGroup && baseGroupBlock !== null && baseGroupBlock !== form.block ? (
                  <Alert
                    type="warning"
                    showIcon
                    message={`В группе «${baseGroup.name}» блокировка по ботам ${baseGroupBlock}, в форме — ${form.block}`}
                    description={
                      'Оптимизатор посчитает базу с блокировкой из формы, поэтому её P&L и МПУ будут отличаться от ' +
                      'тех, что показывает страница группы.'
                    }
                    action={
                      <Button size="small" disabled={formLocked} onClick={() => updateForm({ block: baseGroupBlock })}>
                        Взять из группы
                      </Button>
                    }
                  />
                ) : null}
                {groupMatch && loadedGroup ? (
                  <Alert
                    type={
                      groupMatch.missing.length > 0 ||
                      groupMatch.warnings.length > 0 ||
                      loadedGroup.result.failed.length > 0
                        ? 'warning'
                        : 'success'
                    }
                    showIcon
                    message={`В группе «${loadedGroup.result.groupName}» найдено ${basket.length - groupMatch.missing.length} из ${basket.length} монет корзины`}
                    description={
                      <Flex vertical gap={4}>
                        {groupMatch.missing.length > 0 ? (
                          <Text>Будут посчитаны бэктестами: {groupMatch.missing.join(', ')}.</Text>
                        ) : null}
                        {groupMatch.outsideBasket > 0 ? (
                          <Text type="secondary">
                            Бэктестов с монетами вне корзины: {groupMatch.outsideBasket} — не используются.
                          </Text>
                        ) : null}
                        {loadedGroup.result.failed.length > 0 ? (
                          <Text type="secondary">
                            Не загрузились бэктесты группы:{' '}
                            {loadedGroup.result.failed.map((item) => `#${item.backtestId}`).join(', ')}.
                          </Text>
                        ) : null}
                        {groupMatch.warnings.map((warning) => (
                          <Text key={warning}>{warning}</Text>
                        ))}
                      </Flex>
                    }
                  />
                ) : null}
              </Flex>
            </Card>
          </Flex>
        </Col>

        <Col xs={24} xl={12}>
          <Flex vertical gap={16}>
            <Card title="3. Период и комиссии" size="small">
              <Row gutter={[12, 12]}>
                <Col xs={24} sm={12}>
                  <FormField label="Начало периода">
                    <Input
                      type="date"
                      value={form.periodFrom}
                      disabled={formLocked}
                      onChange={(event) => updateForm({ periodFrom: event.target.value })}
                      aria-label="Начало периода"
                    />
                  </FormField>
                </Col>
                <Col xs={24} sm={12}>
                  <FormField label="Конец периода">
                    <Input
                      type="date"
                      value={form.periodTo}
                      disabled={formLocked}
                      onChange={(event) => updateForm({ periodTo: event.target.value })}
                      aria-label="Конец периода"
                    />
                  </FormField>
                </Col>
                <Col xs={12} sm={8}>
                  <FormField label="Мейкер, %">
                    <InputNumber
                      style={{ width: '100%' }}
                      min={FORM_LIMITS.commission.min}
                      max={FORM_LIMITS.commission.max}
                      step={0.005}
                      value={form.makerCommission}
                      disabled={formLocked}
                      onChange={whenNumber((value) => updateForm({ makerCommission: value }))}
                      aria-label="Комиссия мейкера"
                    />
                  </FormField>
                </Col>
                <Col xs={12} sm={8}>
                  <FormField label="Тейкер, %">
                    <InputNumber
                      style={{ width: '100%' }}
                      min={FORM_LIMITS.commission.min}
                      max={FORM_LIMITS.commission.max}
                      step={0.005}
                      value={form.takerCommission}
                      disabled={formLocked}
                      onChange={whenNumber((value) => updateForm({ takerCommission: value }))}
                      aria-label="Комиссия тейкера"
                    />
                  </FormField>
                </Col>
                <Col xs={24} sm={8}>
                  <FormField label="Учитывать тени">
                    <Switch
                      checked={form.includeWicks}
                      disabled={formLocked}
                      onChange={(checked) => updateForm({ includeWicks: checked })}
                      aria-label="Учитывать тени"
                    />
                  </FormField>
                </Col>
                <Col xs={24} sm={12}>
                  <FormField
                    label="Блокировка по ботам"
                    hint="Максимум одновременно открытых позиций при сложении бэктестов корзины — как настройка на странице группы бэктестов."
                    extra="Блокировка по ботам, как в группе"
                  >
                    <InputNumber
                      style={{ width: '100%' }}
                      min={FORM_LIMITS.block.min}
                      max={FORM_LIMITS.block.max}
                      precision={0}
                      value={form.block}
                      disabled={formLocked}
                      onChange={whenNumber((value) => updateForm({ block: value }))}
                      aria-label="Блокировка по ботам"
                    />
                  </FormField>
                </Col>
              </Row>
              {runSettings.periodFrom ? null : (
                <Alert
                  style={{ marginTop: 12 }}
                  type="error"
                  showIcon
                  message="Некорректный период: дата начала должна быть не позже даты окончания."
                />
              )}
            </Card>

            <Card title="4. Цель" size="small">
              <Flex vertical gap={12}>
                <FormField label="Что максимизируем">
                  <Select<ObjectiveKind>
                    value={form.objective.kind}
                    options={OBJECTIVE_OPTIONS}
                    disabled={formLocked}
                    onChange={(kind) => updateObjective({ kind })}
                    aria-label="Целевая метрика"
                  />
                </FormField>
                <Text type="secondary">{OBJECTIVE_HINTS[form.objective.kind]}</Text>
                <Row gutter={[12, 12]}>
                  {form.objective.kind === 'robust' ? (
                    <Col xs={24} sm={12}>
                      <FormField
                        label="Под-периодов"
                        hint="Период делится на равные части по времени входа в сделку; учитывается худшая часть."
                      >
                        <InputNumber
                          style={{ width: '100%' }}
                          min={FORM_LIMITS.folds.min}
                          max={FORM_LIMITS.folds.max}
                          precision={0}
                          value={form.objective.folds}
                          disabled={formLocked}
                          onChange={whenNumber((value) => updateObjective({ folds: value }))}
                          aria-label="Под-периодов"
                        />
                      </FormField>
                    </Col>
                  ) : null}
                  {form.objective.kind === 'profitCapped' ? (
                    <Col xs={24} sm={12}>
                      <FormField label={`Ограничение МПУ, ${quoteCurrency}`} hint="Пусто — без ограничения.">
                        <InputNumber
                          style={{ width: '100%' }}
                          min={FORM_LIMITS.maxMaeCap.min}
                          max={FORM_LIMITS.maxMaeCap.max}
                          placeholder="без ограничения"
                          value={form.objective.maxMaeCap}
                          disabled={formLocked}
                          onChange={(value) => updateObjective({ maxMaeCap: value ?? null })}
                          aria-label="Ограничение МПУ"
                        />
                      </FormField>
                    </Col>
                  ) : null}
                  <Col xs={24} sm={12}>
                    <FormField
                      label="Минимум сделок"
                      hint="Если сделок на корзине меньше, score умножается на «сделки / минимум»; 0 — без штрафа."
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.minDeals.min}
                        max={FORM_LIMITS.minDeals.max}
                        precision={0}
                        value={form.objective.minDeals}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateObjective({ minDeals: value }))}
                        aria-label="Минимум сделок"
                      />
                    </FormField>
                  </Col>
                </Row>
              </Flex>
            </Card>

            <Card title="5. Поиск и лимиты" size="small">
              <Flex vertical gap={12}>
                <Row gutter={[12, 12]}>
                  <Col xs={12} md={8}>
                    <FormField label="Поколений">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.generations.min}
                        max={FORM_LIMITS.generations.max}
                        precision={0}
                        value={form.search.generations}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ generations: value }))}
                        aria-label="Поколений"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="Кандидатов в поколении" hint="Каждый проверяется на отборочной корзине.">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.candidatesPerGeneration.min}
                        max={FORM_LIMITS.candidatesPerGeneration.max}
                        precision={0}
                        value={form.search.candidatesPerGeneration}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ candidatesPerGeneration: value }))}
                        aria-label="Кандидатов в поколении"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="На полную корзину" hint="Сколько лучших по отбору досчитываются на всей корзине.">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.promotePerGeneration.min}
                        max={FORM_LIMITS.promotePerGeneration.max}
                        precision={0}
                        value={form.search.promotePerGeneration}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ promotePerGeneration: value }))}
                        aria-label="На полную корзину"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="Изменений: от" hint="Сколько параметров меняет одна мутация.">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.mutations.min}
                        max={FORM_LIMITS.mutations.max}
                        precision={0}
                        value={form.search.mutationsMin}
                        disabled={formLocked}
                        onChange={whenNumber((value) =>
                          updateSearch({
                            mutationsMin: value,
                            mutationsMax: Math.max(value, form.search.mutationsMax),
                          }),
                        )}
                        aria-label="Изменений от"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="Изменений: до">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={form.search.mutationsMin}
                        max={FORM_LIMITS.mutations.max}
                        precision={0}
                        value={form.search.mutationsMax}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ mutationsMax: value }))}
                        aria-label="Изменений до"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="Лимит бэктестов" hint="Максимум бэктестов, созданных за весь запуск.">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.maxBacktests.min}
                        max={FORM_LIMITS.maxBacktests.max}
                        precision={0}
                        value={form.search.maxBacktests}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ maxBacktests: value }))}
                        aria-label="Лимит бэктестов"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField
                      label="Допуск отбора, %"
                      hint="Кандидат идёт на полную корзину, если его score отбора не хуже лучшего более чем на допуск."
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.screenTolerance.min * 100}
                        max={FORM_LIMITS.screenTolerance.max * 100}
                        step={1}
                        value={roundTo(form.search.screenTolerance * 100, 2)}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ screenTolerance: value / 100 }))}
                        aria-label="Допуск отбора"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField
                      label="Мин. улучшение, %"
                      hint="Новый лучший принимается, если его score на полной корзине выше лучшего хотя бы на столько."
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.minImprovement.min * 100}
                        max={FORM_LIMITS.minImprovement.max * 100}
                        step={0.5}
                        value={roundTo(form.search.minImprovement * 100, 2)}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ minImprovement: value / 100 }))}
                        aria-label="Минимальное улучшение"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="Зерно поиска" hint="Одинаковое зерно и настройки дают одинаковых кандидатов.">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.seed.min}
                        max={FORM_LIMITS.seed.max}
                        precision={0}
                        value={form.search.seed}
                        disabled={formLocked}
                        onChange={whenNumber((value) => updateSearch({ seed: value }))}
                        aria-label="Зерно поиска"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField
                      label="Потоков"
                      hint="Одновременно выполняющихся бэктестов — потоки тарифа Veles. Меняется на ходу."
                      extra="потоки тарифа Veles"
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.maxInFlight.min}
                        max={FORM_LIMITS.maxInFlight.max}
                        precision={0}
                        value={schedulerValues.maxInFlight}
                        onChange={whenNumber((value) => handleSchedulerChange({ maxInFlight: value }))}
                        aria-label="Потоков"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField
                      label="Интервал создания, с"
                      hint="Пауза между созданием бэктестов. Меньше 31 с — риск ограничения запросов (429). Меняется на ходу."
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.submitDelaySeconds.min}
                        max={FORM_LIMITS.submitDelaySeconds.max}
                        value={schedulerValues.submitDelaySeconds}
                        onChange={whenNumber((value) => handleSchedulerChange({ submitDelaySeconds: value }))}
                        aria-label="Интервал создания бэктестов"
                      />
                    </FormField>
                  </Col>
                  <Col xs={12} md={8}>
                    <FormField label="Таймаут бэктеста, мин">
                      <InputNumber
                        style={{ width: '100%' }}
                        min={FORM_LIMITS.backtestTimeoutMinutes.min}
                        max={FORM_LIMITS.backtestTimeoutMinutes.max}
                        value={schedulerValues.backtestTimeoutMinutes}
                        onChange={whenNumber((value) => handleSchedulerChange({ backtestTimeoutMinutes: value }))}
                        aria-label="Таймаут бэктеста"
                      />
                    </FormField>
                  </Col>
                </Row>
                <Card size="small" type="inner" title="Оценка объёма">
                  <Row gutter={[12, 12]}>
                    <Col xs={12} md={6}>
                      <Statistic title="За поколение, до" value={estimate.perGenerationMax} />
                    </Col>
                    <Col xs={12} md={6}>
                      <Statistic
                        title="База"
                        value={baseGroup && !groupMatch ? 'из группы' : estimate.baseBacktests}
                        valueStyle={baseGroup && !groupMatch ? { fontSize: 16 } : undefined}
                      />
                    </Col>
                    <Col xs={12} md={6}>
                      <Statistic title="Всего, до" value={estimate.totalMax} />
                    </Col>
                    <Col xs={12} md={6}>
                      <Statistic
                        title="Время, не меньше"
                        value={formatDuration(estimate.hoursAtDelay * 3_600_000)}
                        valueStyle={{ fontSize: 16 }}
                      />
                    </Col>
                  </Row>
                  <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
                    Время — только паузы между созданием бэктестов ({form.scheduler.submitDelaySeconds} с); реальная
                    длительность зависит от скорости бэктестов Veles и числа потоков. Повторные бэктесты одних и тех же
                    настроек берутся из кеша.
                  </Text>
                </Card>
              </Flex>
            </Card>
          </Flex>
        </Col>
      </Row>

      <Card title="6. Параметры" size="small" extra={loaded ? <Text type="secondary">{loaded.name}</Text> : undefined}>
        <ParamsTable
          params={params}
          disabled={formLocked}
          onChange={setParams}
          onReset={() => setParams(loaded ? loaded.defaults.map((param) => ({ ...param })) : [])}
        />
      </Card>

      <Card title="7. Запуск" size="small">
        <Flex vertical gap={16}>
          <Flex gap={8} wrap>
            {active ? null : (
              <Button
                type="primary"
                size="large"
                icon={<PlayCircleOutlined />}
                disabled={problems.length > 0 || !extensionReady}
                onClick={() => void handleStart()}
              >
                Старт
              </Button>
            )}
            {active && runStatus === 'running' ? (
              <Button size="large" icon={<PauseCircleOutlined />} onClick={() => portfolioRunSession.pause()}>
                Пауза
              </Button>
            ) : null}
            {active && runStatus === 'paused' ? (
              <Button
                type="primary"
                size="large"
                icon={<PlayCircleOutlined />}
                onClick={() => portfolioRunSession.unpause()}
              >
                Продолжить
              </Button>
            ) : null}
            {active && !stopping ? (
              <Popconfirm
                title="Остановить запуск?"
                description="Созданные бэктесты сохранятся — запуск можно будет продолжить позже."
                okText="Остановить"
                cancelText="Отмена"
                onConfirm={() => portfolioRunSession.stop()}
              >
                <Button danger size="large" icon={<StopOutlined />}>
                  Стоп
                </Button>
              </Popconfirm>
            ) : null}
            {stopping ? (
              <Button size="large" loading disabled>
                Завершение…
              </Button>
            ) : null}
          </Flex>
          {!active && problems.length > 0 ? (
            <Alert
              type="info"
              showIcon
              message="Перед стартом"
              description={
                <ul style={{ margin: 0, paddingInlineStart: 20 }}>
                  {problems.map((problem) => (
                    <li key={problem}>{problem}</li>
                  ))}
                </ul>
              }
            />
          ) : null}
          {active ? (
            <Text type="secondary">
              Можно переходить на другие страницы расширения — запуск продолжится. Пауза не создаёт новых бэктестов, но
              досчитывает уже запущенные; после остановки или перезагрузки запуск можно продолжить.
            </Text>
          ) : null}
          {snapshot ? (
            <RunProgressPanel progress={snapshot.progress} expectedBacktests={runExpectedBacktests} now={now} />
          ) : (
            <Text type="secondary">Запуск ещё не выполнялся.</Text>
          )}
        </Flex>
      </Card>

      <BestCandidateCard
        summary={bestSummary}
        botName={resultsBotName}
        quoteCurrency={resultsQuote}
        canCreateBot={extensionReady}
        isCreatingBot={isCreatingBot}
        onCreateGroup={handleCreateGroup}
        onCopyJson={(target) => void handleCopyJson(target)}
        onCreateBot={(target) => void handleCreateBot(target)}
      />

      <Card title="8. Кандидаты" size="small">
        <CandidatesTable
          rows={rows}
          botName={resultsBotName}
          quoteCurrency={resultsQuote}
          canCreateBot={extensionReady}
          isCreatingBot={isCreatingBot}
          onCreateGroup={handleCreateGroup}
          onCopyJson={(target) => void handleCopyJson(target)}
          onCreateBot={(target) => void handleCreateBot(target)}
        />
      </Card>

      <Card title="9. Журнал" size="small">
        <PortfolioLogViewer logs={snapshot?.logs ?? []} />
      </Card>
    </section>
  );
};

export default PortfolioOptimizerPage;
