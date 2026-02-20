/**
 * Страница полного автопереборщика стратегий
 *
 * Каскадная оптимизация (индикаторы → сетка → TP/SL) +
 * Walk-forward валидация + автоматическое создание бота.
 */

import {
  CheckCircleOutlined,
  CopyOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  RocketOutlined,
  StopOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Flex,
  Input,
  InputNumber,
  message,
  Modal,
  Progress,
  Result,
  Row,
  Select,
  Slider,
  Space,
  Statistic,
  Steps,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchApiKeys } from '../api/apiKeys';
import { fetchBots } from '../api/bots';
import PageHeader from '../components/ui/PageHeader';
import { buildCabinetUrl } from '../lib/cabinetUrls';
import { genomeToStrategy } from '../lib/genomeConverter';
import { AutoOptimizer } from '../services/autoOptimizer';
import type {
  AutoOptimizerCallbacks,
  AutoOptimizerConfig,
  AutoOptimizerLogEntry,
  AutoOptimizerPhase,
  AutoOptimizerProgress,
  AutoOptimizerResult,
  CascadeConfig,
  WalkForwardConfig,
  WalkForwardResult,
} from '../types/autoOptimizer';
import type { ApiKey } from '../types/apiKeys';
import type { EvaluatedGenome, OptimizationTarget } from '../types/optimizer';
import type { TradingBot } from '../types/bots';

const { Text, Paragraph } = Typography;

interface AutoOptimizerPageProps {
  extensionReady: boolean;
}

// ═══════════════════════════════════════════════════════════════════
// ДЕФОЛТЫ
// ═══════════════════════════════════════════════════════════════════

const DEFAULT_CASCADE: CascadeConfig = {
  indicators: { enabled: true, generations: 6, populationSize: 12 },
  grid: { enabled: true, generations: 8, populationSize: 15 },
  tpSl: { enabled: true, generations: 5, populationSize: 10 },
};

const DEFAULT_WF: Omit<WalkForwardConfig, 'periodFrom' | 'periodTo'> = {
  windowCount: 4,
  trainRatio: 0.75,
  sliding: true,
};

const DEFAULT_GENETIC = {
  mutationRate: 0.35,
  crossoverRate: 0.7,
  elitismCount: 2,
  tournamentSize: 3,
  backtestDelaySeconds: 31,
};

const DEFAULT_TARGET: OptimizationTarget = {
  metric: 'composite' as const,
  weights: { pnl: 0.3, winRate: 0.2, maxDrawdown: 0.2, pnlToRisk: 0.3 },
};

// ═══════════════════════════════════════════════════════════════════
// PHASE STEPS
// ═══════════════════════════════════════════════════════════════════

const PHASE_STEP_MAP: Record<AutoOptimizerPhase, number> = {
  idle: -1,
  loading_strategy: 0,
  phase_indicators: 1,
  phase_grid: 2,
  phase_tp_sl: 3,
  walk_forward_test: 4,
  creating_bot: 5,
  completed: 6,
  error: -1,
};

const STEP_ITEMS = [
  { title: 'Подготовка' },
  { title: 'Индикаторы' },
  { title: 'Сетка DCA' },
  { title: 'TP / SL' },
  { title: 'Walk-Forward' },
  { title: 'Создание бота' },
  { title: 'Готово' },
];

// ═══════════════════════════════════════════════════════════════════
// LOG VIEWER
// ═══════════════════════════════════════════════════════════════════

const LogViewer = ({ logs }: { logs: AutoOptimizerLogEntry[] }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const levelColors: Record<string, string> = {
    info: '#1890ff',
    success: '#52c41a',
    warning: '#faad14',
    error: '#ff4d4f',
  };

  return (
    <div
      ref={containerRef}
      style={{
        maxHeight: 350,
        overflowY: 'auto',
        background: '#1a1a1a',
        padding: 12,
        borderRadius: 8,
        fontFamily: 'monospace',
        fontSize: 12,
      }}
    >
      {logs.length === 0 && <Text type="secondary">Логи появятся после запуска...</Text>}
      {logs.map((log) => (
        <div key={log.id} style={{ marginBottom: 2 }}>
          <Text style={{ color: '#666' }}>{new Date(log.timestamp).toLocaleTimeString()}</Text>{' '}
          <Text style={{ color: levelColors[log.level] }}>{log.message}</Text>
        </div>
      ))}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════
// WF RESULTS TABLE
// ═══════════════════════════════════════════════════════════════════

const WalkForwardTable = ({ results }: { results: WalkForwardResult[] }) => {
  const columns: ColumnsType<WalkForwardResult> = [
    {
      title: '#',
      key: 'idx',
      width: 40,
      render: (_, record) => record.window.index + 1,
    },
    {
      title: 'Train',
      key: 'train',
      render: (_, record) => (
        <Text type="secondary" style={{ fontSize: 11 }}>
          {record.window.trainFrom} → {record.window.trainTo}
        </Text>
      ),
    },
    {
      title: 'Test',
      key: 'test',
      render: (_, record) => (
        <Text style={{ fontSize: 11 }}>
          {record.window.testFrom} → {record.window.testTo}
        </Text>
      ),
    },
    {
      title: 'Train Score',
      key: 'trainScore',
      width: 100,
      render: (_, record) => record.trainScore.toFixed(4),
    },
    {
      title: 'Test Score',
      key: 'testScore',
      width: 100,
      render: (_, record) => (
        <Text type={record.testScore > 0 ? 'success' : 'danger'}>
          {record.testScore.toFixed(4)}
        </Text>
      ),
    },
    {
      title: 'Test PnL',
      key: 'testPnl',
      width: 100,
      render: (_, record) => (
        <Text type={record.testFitness.totalPnl >= 0 ? 'success' : 'danger'}>
          ${record.testFitness.totalPnl.toFixed(2)}
        </Text>
      ),
    },
    {
      title: 'WR',
      key: 'wr',
      width: 60,
      render: (_, record) => `${record.testFitness.winRate.toFixed(1)}%`,
    },
    {
      title: 'Сделки',
      key: 'deals',
      width: 60,
      render: (_, record) => record.testFitness.totalDeals,
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={results}
      rowKey={(r) => String(r.window.index)}
      size="small"
      pagination={false}
    />
  );
};

// ═══════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════

const AutoOptimizerPage = ({ extensionReady }: AutoOptimizerPageProps) => {
  // ── Состояние UI ──
  const [bots, setBots] = useState<TradingBot[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);

  // ── Конфигурация ──
  const [selectedBotId, setSelectedBotId] = useState<number | null>(null);
  const [selectedApiKey, setSelectedApiKey] = useState<number | null>(null);
  const [symbols, setSymbols] = useState('BTC');
  const [periodFrom, setPeriodFrom] = useState('2023-02-16');
  const [periodTo, setPeriodTo] = useState('2026-02-16');
  const [selectedExchange, setSelectedExchange] = useState('BINANCE_FUTURES');
  // NOTE: BYBIT_FUTURES может не поддерживаться для бэктестов в Veles
  const [selectedQuoteCurrency, setSelectedQuoteCurrency] = useState('USDT');
  const [cascade, setCascade] = useState<CascadeConfig>(DEFAULT_CASCADE);
  const [windowCount, setWindowCount] = useState(DEFAULT_WF.windowCount);
  const [trainRatio, setTrainRatio] = useState(DEFAULT_WF.trainRatio);
  const [sliding, setSliding] = useState(DEFAULT_WF.sliding);
  const [backtestDelay, setBacktestDelay] = useState(DEFAULT_GENETIC.backtestDelaySeconds);
  const [autoCreateBot, setAutoCreateBot] = useState(true);
  const [minRobustness, setMinRobustness] = useState(0.05);
  const [minDeals, setMinDeals] = useState(10);
  const [botDeposit, setBotDeposit] = useState(10);
  const [botLeverage, setBotLeverage] = useState(10);
  const [botMarginType, setBotMarginType] = useState<'ISOLATED' | 'CROSS'>('CROSS');

  // ── Рантайм ──
  const [logs, setLogs] = useState<AutoOptimizerLogEntry[]>([]);
  const [progress, setProgress] = useState<AutoOptimizerProgress | null>(null);
  const [wfResults, setWfResults] = useState<WalkForwardResult[]>([]);
  const [finalResult, setFinalResult] = useState<AutoOptimizerResult | null>(null);
  const optimizerRef = useRef<AutoOptimizer | null>(null);

  // ── Загрузка данных ──
  const loadData = useCallback(async () => {
    if (!extensionReady) return;
    setLoading(true);

    // Загружаем независимо, чтобы падение одного не блокировало другой
    const botsPromise = fetchBots({ page: 0, size: 100 })
      .then((res) => {
        setBots(res.content as TradingBot[]);
      })
      .catch((err) => {
        console.error('AutoOptimizer: failed to load bots:', err);
        message.warning('Не удалось загрузить ботов (можно работать без базового бота)');
      });

    const keysPromise = fetchApiKeys()
      .then((keys) => {
        setApiKeys(keys);
        if (keys.length === 0) {
          message.warning('API-ключи не найдены. Создайте API-ключ на veles.finance');
        }
      })
      .catch((err) => {
        console.error('AutoOptimizer: failed to load API keys:', err);
        message.error(`Не удалось загрузить API-ключи: ${err instanceof Error ? err.message : 'Ошибка'}`);
      });

    await Promise.all([botsPromise, keysPromise]);
    setLoading(false);
  }, [extensionReady]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // ── Callbacks ──
  const callbacks: AutoOptimizerCallbacks = useMemo(
    () => ({
      onLog: (level, msg) => {
        setLogs((prev) => [
          ...prev,
          { id: `${Date.now()}-${Math.random()}`, timestamp: Date.now(), level, message: msg },
        ]);
      },
      onProgress: (p) => setProgress(p),
      onPhaseComplete: (_phase, _genome) => {
        // Phase markers автоматом через progress
      },
      onWalkForwardResult: (r) => {
        setWfResults((prev) => [...prev, r]);
      },
      onComplete: (result) => {
        setFinalResult(result);
        setRunning(false);
        setPaused(false);
      },
    }),
    [],
  );

  // ── Запуск ──
  const handleStart = useCallback(async () => {
    if (!selectedApiKey) {
      message.error('Выберите API-ключ');
      return;
    }

    const parsedSymbols = symbols
      .split(/[,\s]+/)
      .map((s) => s.trim().toUpperCase())
      .filter(Boolean);

    if (parsedSymbols.length === 0) {
      message.error('Укажите хотя бы один символ');
      return;
    }

    const config: AutoOptimizerConfig = {
      botId: selectedBotId ?? undefined,
      apiKeyId: selectedApiKey,
      symbols: parsedSymbols,
      exchange: selectedExchange,
      quoteCurrency: selectedQuoteCurrency,
      walkForward: {
        periodFrom: `${periodFrom}T00:00:00.000Z`,
        periodTo: `${periodTo}T23:59:59.999Z`,
        windowCount,
        trainRatio,
        sliding,
      },
      cascade,
      genetic: {
        ...DEFAULT_GENETIC,
        backtestDelaySeconds: backtestDelay,
      },
      target: DEFAULT_TARGET,
      autoCreateBot,
      minRobustnessScore: minRobustness,
      minDealsPerWindow: minDeals,
      botDeposit,
      botLeverage,
      botMarginType,
    };

    setLogs([]);
    setWfResults([]);
    setFinalResult(null);
    setRunning(true);
    setPaused(false);

    const optimizer = new AutoOptimizer(config, callbacks);
    optimizerRef.current = optimizer;

    try {
      await optimizer.start();
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Ошибка';
      message.error(msg);
      setRunning(false);
    }
  }, [
    selectedBotId, selectedApiKey, symbols, periodFrom, periodTo,
    selectedExchange, selectedQuoteCurrency,
    windowCount, trainRatio, sliding, cascade, backtestDelay,
    autoCreateBot, minRobustness, minDeals, botDeposit, botLeverage,
    botMarginType, callbacks,
  ]);

  const handleStop = useCallback(() => {
    optimizerRef.current?.stop();
    setRunning(false);
    setPaused(false);
  }, []);

  const handlePause = useCallback(() => {
    if (paused) {
      optimizerRef.current?.unpause();
      setPaused(false);
    } else {
      optimizerRef.current?.pause();
      setPaused(true);
    }
  }, [paused]);

  // ── Export JSON ──
  const handleExport = useCallback(() => {
    if (!finalResult) return;
    const blob = new Blob([JSON.stringify(finalResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `auto_optimizer_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [finalResult]);

  // ── Estimated time ──
  const estimatedHours = useMemo(() => {
    const symbolCount = symbols.split(/[,\s]+/).filter(Boolean).length || 1;
    let totalBt = 0;
    if (cascade.indicators.enabled) totalBt += cascade.indicators.generations * cascade.indicators.populationSize * symbolCount;
    if (cascade.grid.enabled) totalBt += cascade.grid.generations * cascade.grid.populationSize * symbolCount;
    if (cascade.tpSl.enabled) totalBt += cascade.tpSl.generations * cascade.tpSl.populationSize * symbolCount;
    // WF
    const wfGens = Math.max(3, Math.floor(cascade.grid.generations / 2));
    const wfPop = Math.max(8, Math.floor(cascade.grid.populationSize / 2));
    totalBt += windowCount * (wfGens * wfPop * symbolCount + symbolCount);
    const totalSec = totalBt * backtestDelay;
    return (totalSec / 3600).toFixed(1);
  }, [cascade, windowCount, backtestDelay, symbols]);

  // ── Прогресс % ──
  const progressPercent = useMemo(() => {
    if (!progress || progress.totalBacktests === 0) return 0;
    return Math.round((progress.completedBacktests / progress.totalBacktests) * 100);
  }, [progress]);

  const currentStep = progress ? PHASE_STEP_MAP[progress.phase] ?? -1 : -1;

  if (!extensionReady) {
    return (
      <Alert
        message="Расширение не активно"
        description="Автопереборщик работает только через расширение Veles Tools."
        type="warning"
        showIcon
      />
    );
  }

  return (
    <div style={{ padding: '0 0 40px' }}>
      <PageHeader
        title="Автопереборщик стратегий"
        description="Каскадная оптимизация + Walk-Forward валидация + автосоздание ботов"
        extra={
          <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
            Обновить данные
          </Button>
        }
      />

      {/* ═══════ ПРОГРЕСС ═══════ */}
      {running && (
        <Card style={{ marginBottom: 16 }}>
          <Steps
            current={currentStep}
            items={STEP_ITEMS}
            size="small"
            style={{ marginBottom: 16 }}
          />
          <Progress
            percent={progressPercent}
            status={paused ? 'exception' : 'active'}
            strokeColor={paused ? '#faad14' : undefined}
          />
          <Flex justify="space-between" align="center" style={{ marginTop: 12 }}>
            <Space>
              <Text type="secondary">
                Бэктестов: {progress?.completedBacktests ?? 0} / {progress?.totalBacktests ?? '?'}
              </Text>
              {progress?.estimatedEndAt && (
                <Text type="secondary">
                  ETA: {new Date(progress.estimatedEndAt).toLocaleTimeString()}
                </Text>
              )}
            </Space>
            <Space>
              <Button
                icon={paused ? <PlayCircleOutlined /> : <PauseCircleOutlined />}
                onClick={handlePause}
              >
                {paused ? 'Продолжить' : 'Пауза'}
              </Button>
              <Button danger icon={<StopOutlined />} onClick={handleStop}>
                Стоп
              </Button>
            </Space>
          </Flex>
        </Card>
      )}

      {/* ═══════ РЕЗУЛЬТАТ ═══════ */}
      {finalResult && (
        <Card
          title="🏆 Результат автопереборщика"
          style={{ marginBottom: 16 }}
          extra={
            <Button icon={<DownloadOutlined />} onClick={handleExport}>
              Экспорт JSON
            </Button>
          }
        >
          <Row gutter={16}>
            <Col span={4}>
              <Statistic
                title="Robustness"
                value={finalResult.aggregation.robustnessScore}
                precision={4}
                valueStyle={{
                  color: finalResult.aggregation.robustnessScore > 0 ? '#3f8600' : '#cf1322',
                }}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="Median Test Score"
                value={finalResult.aggregation.medianTestScore}
                precision={4}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="Overfit Ratio"
                value={finalResult.aggregation.overfitRatio}
                precision={2}
                valueStyle={{
                  color: finalResult.aggregation.overfitRatio > 0.5 ? '#3f8600' : '#cf1322',
                }}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="Avg Test PnL"
                value={finalResult.aggregation.avgTestPnl}
                precision={2}
                prefix="$"
                valueStyle={{
                  color: finalResult.aggregation.avgTestPnl >= 0 ? '#3f8600' : '#cf1322',
                }}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="Avg Win Rate"
                value={finalResult.aggregation.avgTestWinRate}
                precision={1}
                suffix="%"
              />
            </Col>
            <Col span={4}>
              <Statistic title="Всего бэктестов" value={finalResult.totalBacktests} />
            </Col>
          </Row>

          {finalResult.createdBotId !== null && (
            <Alert
              type="success"
              style={{ marginTop: 16 }}
              message={
                <Space>
                  <CheckCircleOutlined />
                  <span>Бот создан!</span>
                  <a
                    href={buildCabinetUrl(`bots/${finalResult.createdBotId}`)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Открыть бота #{finalResult.createdBotId}
                  </a>
                </Space>
              }
            />
          )}

          {finalResult.createdBotId === null && finalResult.aggregation.robustnessScore > 0 && (
            <Alert
              type="warning"
              style={{ marginTop: 16 }}
              message="Бот не создан — robustness ниже порога или авто-создание отключено"
            />
          )}

          {/* Walk-Forward таблица */}
          <Card title="Walk-Forward результаты" size="small" style={{ marginTop: 16 }}>
            <WalkForwardTable results={finalResult.walkForwardResults} />
          </Card>

          {/* Лучший геном */}
          <Card title="Лучший геном" size="small" style={{ marginTop: 16 }}>
            <Descriptions column={3} size="small" bordered>
              <Descriptions.Item label="Score">
                {finalResult.bestGenome.fitness.score.toFixed(4)}
              </Descriptions.Item>
              <Descriptions.Item label="PnL">
                ${finalResult.bestGenome.fitness.totalPnl.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="Win Rate">
                {finalResult.bestGenome.fitness.winRate.toFixed(1)}%
              </Descriptions.Item>
              <Descriptions.Item label="Deals">
                {finalResult.bestGenome.fitness.totalDeals}
              </Descriptions.Item>
              <Descriptions.Item label="Algorithm">
                {finalResult.bestGenome.genome.algorithm}
              </Descriptions.Item>
              <Descriptions.Item label="Leverage">
                x{finalResult.bestGenome.genome.leverage}
              </Descriptions.Item>
              <Descriptions.Item label="Условия входа" span={3}>
                <Space wrap>
                  {finalResult.bestGenome.genome.entryConditions.map((c, i) => (
                    <Tag key={i} color="blue">
                      {c.indicator} {c.operation} {c.value ?? ''} ({c.interval})
                    </Tag>
                  ))}
                </Space>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Card>
      )}

      {/* ═══════ КОНФИГУРАЦИЯ ═══════ */}
      {!running && (
        <Row gutter={16}>
          {/* Левая колонка: основное */}
          <Col span={12}>
            <Card title="🤖 Базовый бот (опционально)" size="small" style={{ marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text type="secondary">Бот-источник стратегии (пусто = с нуля):</Text>
                <Select
                  showSearch
                  allowClear
                  placeholder="Без бота — генерация с нуля"
                  value={selectedBotId}
                  onChange={setSelectedBotId}
                  style={{ width: '100%' }}
                  optionFilterProp="label"
                  options={bots.map((b) => ({
                    value: b.id,
                    label: `#${b.id} — ${b.name}`,
                  }))}
                />
                {!selectedBotId && (
                  <Alert
                    type="info"
                    showIcon
                    message="Режим «с нуля»: стратегии будут генерироваться случайно"
                    style={{ marginTop: 4 }}
                  />
                )}
                <Row gutter={12} style={{ marginTop: selectedBotId ? 0 : 8 }}>
                  <Col span={12}>
                    <Text type="secondary">Биржа:</Text>
                    <Select
                      value={selectedExchange}
                      onChange={setSelectedExchange}
                      style={{ width: '100%' }}
                      disabled={!!selectedBotId}
                      options={[
                        { value: 'BINANCE_FUTURES', label: 'Binance Futures' },
                        { value: 'BYBIT_FUTURES', label: 'Bybit Futures' },
                        { value: 'OKX_FUTURES', label: 'OKX Futures' },
                      ]}
                    />
                  </Col>
                  <Col span={12}>
                    <Text type="secondary">Quote-валюта:</Text>
                    <Select
                      value={selectedQuoteCurrency}
                      onChange={setSelectedQuoteCurrency}
                      style={{ width: '100%' }}
                      disabled={!!selectedBotId}
                      options={[
                        { value: 'USDT', label: 'USDT' },
                        { value: 'USDC', label: 'USDC' },
                        { value: 'BUSD', label: 'BUSD' },
                      ]}
                    />
                  </Col>
                </Row>
                <Text type="secondary">
                  API-ключ ({apiKeys.length > 0 ? `найдено: ${apiKeys.length}` : 'не найдены — обновите данные'}):
                </Text>
                <Space.Compact style={{ width: '100%' }}>
                  <Select
                    placeholder={loading ? 'Загрузка...' : 'Выберите API-ключ'}
                    value={selectedApiKey}
                    onChange={setSelectedApiKey}
                    style={{ width: '100%' }}
                    loading={loading}
                    notFoundContent={loading ? 'Загрузка...' : 'API-ключи не найдены. Убедитесь что связь активна и нажмите «Обновить»'}
                    options={apiKeys.map((k) => ({
                      value: k.id,
                      label: `${k.name} (${k.exchange})`,
                    }))}
                  />
                  <Button onClick={loadData} loading={loading}>
                    Обновить
                  </Button>
                </Space.Compact>
              </Space>
            </Card>

            <Card title="📅 Период и символы" size="small" style={{ marginBottom: 16 }}>
              <Row gutter={12}>
                <Col span={12}>
                  <Text type="secondary">Начало:</Text>
                  <Input
                    value={periodFrom}
                    onChange={(e) => setPeriodFrom(e.target.value)}
                    placeholder="2023-02-16"
                  />
                </Col>
                <Col span={12}>
                  <Text type="secondary">Конец:</Text>
                  <Input
                    value={periodTo}
                    onChange={(e) => setPeriodTo(e.target.value)}
                    placeholder="2026-02-16"
                  />
                </Col>
              </Row>
              <div style={{ marginTop: 12 }}>
                <Text type="secondary">Символы (через запятую):</Text>
                <Input
                  value={symbols}
                  onChange={(e) => setSymbols(e.target.value)}
                  placeholder="BTC, ETH, SOL"
                />
              </div>
            </Card>

            <Card title="🧬 Каскадная оптимизация" size="small" style={{ marginBottom: 16 }}>
              {/* Phase 1: Indicators */}
              <Flex align="center" gap={8} style={{ marginBottom: 8 }}>
                <Checkbox
                  checked={cascade.indicators.enabled}
                  onChange={(e) =>
                    setCascade((prev) => ({
                      ...prev,
                      indicators: { ...prev.indicators, enabled: e.target.checked },
                    }))
                  }
                />
                <Text strong>Фаза 1: Индикаторы</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={30}
                  value={cascade.indicators.generations}
                  onChange={(v) =>
                    setCascade((prev) => ({
                      ...prev,
                      indicators: { ...prev.indicators, generations: v ?? 6 },
                    }))
                  }
                  addonBefore="Поколений"
                  style={{ width: 150 }}
                />
                <InputNumber
                  size="small"
                  min={4}
                  max={50}
                  value={cascade.indicators.populationSize}
                  onChange={(v) =>
                    setCascade((prev) => ({
                      ...prev,
                      indicators: { ...prev.indicators, populationSize: v ?? 12 },
                    }))
                  }
                  addonBefore="Особей"
                  style={{ width: 140 }}
                />
              </Flex>

              {/* Phase 2: Grid */}
              <Flex align="center" gap={8} style={{ marginBottom: 8 }}>
                <Checkbox
                  checked={cascade.grid.enabled}
                  onChange={(e) =>
                    setCascade((prev) => ({
                      ...prev,
                      grid: { ...prev.grid, enabled: e.target.checked },
                    }))
                  }
                />
                <Text strong>Фаза 2: Сетка DCA</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={30}
                  value={cascade.grid.generations}
                  onChange={(v) =>
                    setCascade((prev) => ({
                      ...prev,
                      grid: { ...prev.grid, generations: v ?? 8 },
                    }))
                  }
                  addonBefore="Поколений"
                  style={{ width: 150 }}
                />
                <InputNumber
                  size="small"
                  min={4}
                  max={50}
                  value={cascade.grid.populationSize}
                  onChange={(v) =>
                    setCascade((prev) => ({
                      ...prev,
                      grid: { ...prev.grid, populationSize: v ?? 15 },
                    }))
                  }
                  addonBefore="Особей"
                  style={{ width: 140 }}
                />
              </Flex>

              {/* Phase 3: TP/SL */}
              <Flex align="center" gap={8}>
                <Checkbox
                  checked={cascade.tpSl.enabled}
                  onChange={(e) =>
                    setCascade((prev) => ({
                      ...prev,
                      tpSl: { ...prev.tpSl, enabled: e.target.checked },
                    }))
                  }
                />
                <Text strong>Фаза 3: TP / SL</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={30}
                  value={cascade.tpSl.generations}
                  onChange={(v) =>
                    setCascade((prev) => ({
                      ...prev,
                      tpSl: { ...prev.tpSl, generations: v ?? 5 },
                    }))
                  }
                  addonBefore="Поколений"
                  style={{ width: 150 }}
                />
                <InputNumber
                  size="small"
                  min={4}
                  max={50}
                  value={cascade.tpSl.populationSize}
                  onChange={(v) =>
                    setCascade((prev) => ({
                      ...prev,
                      tpSl: { ...prev.tpSl, populationSize: v ?? 10 },
                    }))
                  }
                  addonBefore="Особей"
                  style={{ width: 140 }}
                />
              </Flex>
            </Card>
          </Col>

          {/* Правая колонка */}
          <Col span={12}>
            <Card title="🔄 Walk-Forward валидация" size="small" style={{ marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Flex align="center" gap={16}>
                  <div>
                    <Text type="secondary">Окон:</Text>
                    <InputNumber
                      min={2}
                      max={10}
                      value={windowCount}
                      onChange={(v) => setWindowCount(v ?? 4)}
                      style={{ width: 80, marginLeft: 8 }}
                    />
                  </div>
                  <div>
                    <Text type="secondary">Train доля:</Text>
                    <InputNumber
                      min={0.5}
                      max={0.95}
                      step={0.05}
                      value={trainRatio}
                      onChange={(v) => setTrainRatio(v ?? 0.75)}
                      style={{ width: 80, marginLeft: 8 }}
                    />
                  </div>
                  <Checkbox checked={sliding} onChange={(e) => setSliding(e.target.checked)}>
                    Скользящее
                  </Checkbox>
                </Flex>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  Walk-forward разбивает период на {windowCount} окон. На каждом: обучение ({Math.round(trainRatio * 100)}%) →
                  тест ({Math.round((1 - trainRatio) * 100)}%). Это защищает от оверфиттинга.
                </Text>
              </Space>
            </Card>

            <Card title="🤖 Автосоздание бота" size="small" style={{ marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Checkbox checked={autoCreateBot} onChange={(e) => setAutoCreateBot(e.target.checked)}>
                  Автоматически создать бота из лучшего результата
                </Checkbox>
                <Row gutter={12}>
                  <Col span={8}>
                    <Text type="secondary">Мин. robustness:</Text>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.01}
                      value={minRobustness}
                      onChange={(v) => setMinRobustness(v ?? 0.05)}
                      style={{ width: '100%' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Text type="secondary">Мин. сделок/окно:</Text>
                    <InputNumber
                      size="small"
                      min={1}
                      max={100}
                      value={minDeals}
                      onChange={(v) => setMinDeals(v ?? 10)}
                      style={{ width: '100%' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Text type="secondary">Депозит $:</Text>
                    <InputNumber
                      size="small"
                      min={1}
                      max={10000}
                      value={botDeposit}
                      onChange={(v) => setBotDeposit(v ?? 10)}
                      style={{ width: '100%' }}
                    />
                  </Col>
                </Row>
                <Row gutter={12}>
                  <Col span={8}>
                    <Text type="secondary">Плечо:</Text>
                    <InputNumber
                      size="small"
                      min={1}
                      max={125}
                      value={botLeverage}
                      onChange={(v) => setBotLeverage(v ?? 10)}
                      style={{ width: '100%' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Text type="secondary">Маржа:</Text>
                    <Select
                      size="small"
                      value={botMarginType}
                      onChange={setBotMarginType}
                      style={{ width: '100%' }}
                      options={[
                        { value: 'CROSS', label: 'Cross' },
                        { value: 'ISOLATED', label: 'Isolated' },
                      ]}
                    />
                  </Col>
                </Row>
              </Space>
            </Card>

            <Card title="⚙️ Параметры" size="small" style={{ marginBottom: 16 }}>
              <Flex align="center" gap={16}>
                <div>
                  <Text type="secondary">Задержка между бэктестами (сек):</Text>
                  <Slider
                    min={3}
                    max={60}
                    value={backtestDelay}
                    onChange={setBacktestDelay}
                    style={{ width: 200 }}
                    marks={{ 3: '3', 31: '31', 60: '60' }}
                  />
                </div>
              </Flex>
              <Alert
                type="info"
                message={`Ориентировочное время: ~${estimatedHours} часов`}
                style={{ marginTop: 8 }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* ═══════ КНОПКА ЗАПУСКА ═══════ */}
      {!running && !finalResult && (
        <Flex justify="center" style={{ marginTop: 16 }}>
          <Button
            type="primary"
            size="large"
            icon={<RocketOutlined />}
            onClick={handleStart}
            loading={loading}
            disabled={!selectedApiKey}
          >
            Запустить автопереборщик
          </Button>
        </Flex>
      )}

      {!running && finalResult && (
        <Flex justify="center" gap={12} style={{ marginTop: 16 }}>
          <Button
            type="primary"
            size="large"
            icon={<RocketOutlined />}
            onClick={() => {
              setFinalResult(null);
              setLogs([]);
              setWfResults([]);
            }}
          >
            Новый запуск
          </Button>
        </Flex>
      )}

      {/* ═══════ ЛОГИ ═══════ */}
      {logs.length > 0 && (
        <Card title="📋 Логи" size="small" style={{ marginTop: 16 }}>
          <LogViewer logs={logs} />
        </Card>
      )}

      {/* ═══════ WF RESULTS (live) ═══════ */}
      {wfResults.length > 0 && !finalResult && (
        <Card title="Walk-Forward (в процессе)" size="small" style={{ marginTop: 16 }}>
          <WalkForwardTable results={wfResults} />
        </Card>
      )}
    </div>
  );
};

export default AutoOptimizerPage;
