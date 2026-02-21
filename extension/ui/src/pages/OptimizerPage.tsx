/**
 * Страница генетического оптимизатора стратегий
 */

import {
  CopyOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  SettingOutlined,
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
  Row,
  Select,
  Slider,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchBots } from '../api/bots';
import PageHeader from '../components/ui/PageHeader';
import { buildCabinetUrl } from '../lib/cabinetUrls';
import { genomeToStrategy } from '../lib/genomeConverter';
import { dtoToGridOrderGene } from '../lib/genomeConverter';
import OrderSettingsModal from '../components/OrderSettingsModal';
import {
  CATEGORY_LABELS,
  INDICATORS_BY_CATEGORY,
} from '../lib/indicatorCatalog';
import {
  createOptimizerConfig,
  GeneticOptimizer,
  getSavedOptimizerInfo,
  getSavedTopGenomes,
  hasSavedOptimizerState,
  parseSymbols,
  type OptimizerCallbacks,
  type OptimizationRunConfig,
} from '../services/optimizer';
import type {
  EvaluatedGenome,
  GeneticConfig,
  IndicatorCategory,
  OptimizationLogEntry,
  OptimizationProgress,
  OptimizationScope,
  OptimizationStatus,
  OptimizationTarget,
  OrderOptimizationConfig,
} from '../types/optimizer';
import type { TradingBot } from '../types/bots';

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

interface OptimizerPageProps {
  extensionReady: boolean;
}

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

const DEFAULT_GENETIC_CONFIG: GeneticConfig = {
  populationSize: 20,
  generations: 10,
  mutationRate: 0.3,
  crossoverRate: 0.7,
  elitismCount: 2,
  tournamentSize: 3,
  backtestDelaySeconds: 31, // 31 сек по умолчанию
};

const DEFAULT_SCOPE: OptimizationScope = {
  entryConditions: false, // Общий флаг - выключен, мутируем только числа
  entryConditionValues: false, // Пороговые значения - выключено (экспериментально)
  entryConditionIndicators: false, // Сами индикаторы - выключено (экспериментально)
  dcaConditions: false, // Условия в DCA ордерах - выключено (экспериментально)
  dcaStructure: false,
  dcaIndents: true, // ✅ Отступы - главное что оптимизируем
  dcaVolumes: true, // ✅ Объёмы - главное что оптимизируем
  takeProfit: true, // ✅ Значение TP - оптимизируем
  takeProfitIndicator: false, // Индикатор TP - выключено
  stopLoss: false,
  leverage: false,
};

const DEFAULT_TARGET: OptimizationTarget = {
  metric: 'pnlToRisk',
  weights: {
    pnl: 0.3,
    winRate: 0.2,
    maxDrawdown: 0.2,
    pnlToRisk: 0.3,
  },
};

// ═══════════════════════════════════════════════════════════════════
// КОМПОНЕНТЫ
// ═══════════════════════════════════════════════════════════════════

const IndicatorCatalogView = () => {
  const categories = Object.keys(INDICATORS_BY_CATEGORY) as IndicatorCategory[];

  return (
    <Card title="📊 Каталог индикаторов Veles" size="small">
      <Tabs
        items={categories.map((cat) => ({
          key: cat,
          label: CATEGORY_LABELS[cat],
          children: (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {INDICATORS_BY_CATEGORY[cat].map((ind) => (
                <Tag key={ind.id} color={ind.hasValue ? 'blue' : 'green'}>
                  {ind.nameRu}
                  {ind.hasValue && ind.defaultValue !== null && ` (${ind.defaultValue})`}
                </Tag>
              ))}
            </div>
          ),
        }))}
      />
      <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
        <Tag color="blue">синие</Tag> — с числовым порогом,{' '}
        <Tag color="green">зелёные</Tag> — канальные/кроссы
      </Paragraph>
    </Card>
  );
};

interface LogViewerProps {
  logs: OptimizationLogEntry[];
}

const LogViewer = ({ logs }: LogViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Автопрокрутка вниз при новых логах
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
        maxHeight: 300,
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
        <div key={log.id} style={{ marginBottom: 4 }}>
          <Text style={{ color: '#888' }}>
            {new Date(log.timestamp).toLocaleTimeString()}
          </Text>{' '}
          <Text style={{ color: levelColors[log.level] }}>{log.message}</Text>
        </div>
      ))}
    </div>
  );
};

interface TopGenomesTableProps {
  genomes: EvaluatedGenome[];
  onSelect?: (genome: EvaluatedGenome) => void;
}

const TopGenomesTable = ({ genomes, onSelect }: TopGenomesTableProps) => {
  const columns: ColumnsType<EvaluatedGenome> = [
    {
      title: '#',
      key: 'rank',
      width: 50,
      render: (_, __, index) => index + 1,
    },
    {
      title: 'Score',
      dataIndex: ['fitness', 'score'],
      key: 'score',
      width: 80,
      render: (value: number, record: EvaluatedGenome) => (
        <span>
          {value.toFixed(3)}
          {record.paretoOptimal && <span style={{ color: '#faad14', marginLeft: 4 }}>★</span>}
        </span>
      ),
    },
    {
      title: 'PnL',
      dataIndex: ['fitness', 'totalPnl'],
      key: 'pnl',
      width: 100,
      render: (value: number) => (
        <Text type={value >= 0 ? 'success' : 'danger'}>
          {value >= 0 ? '+' : ''}${value.toFixed(2)}
        </Text>
      ),
    },
    {
      title: 'Win%',
      dataIndex: ['fitness', 'winRate'],
      key: 'winRate',
      width: 80,
      render: (value: number) => `${value.toFixed(1)}%`,
    },
    {
      title: 'DD',
      dataIndex: ['fitness', 'maxDrawdown'],
      key: 'drawdown',
      width: 80,
      render: (value: number) => <Text type="danger">{value.toFixed(1)}%</Text>,
    },

    {
      title: 'Условия',
      key: 'conditions',
      render: (_, record) => (
        <Text type="secondary">{record.genome.entryConditions.length} инд.</Text>
      ),
    },
    {
      title: 'DCA',
      key: 'dca',
      width: 60,
      render: (_, record) => record.genome.dcaOrders.length,
    },
    {
      title: 'Поколение',
      dataIndex: ['genome', 'generation'],
      key: 'generation',
      width: 80,
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={genomes}
      rowKey={(record) => record.genome.id}
      size="small"
      pagination={false}
      scroll={{ y: 300 }}
      onRow={(record) => ({
        onClick: () => onSelect?.(record),
        style: { cursor: onSelect ? 'pointer' : 'default' },
      })}
    />
  );
};

/**
 * Округление числа для отображения
 */
const roundForDisplay = (value: number): string => {
  return (Math.round(value * 100) / 100).toString();
};

interface GenomeDetailsModalProps {
  genome: EvaluatedGenome | null;
  open: boolean;
  onClose: () => void;
  onExport: (genome: EvaluatedGenome) => void;
}

const GenomeDetailsModal = ({ genome, open, onClose, onExport }: GenomeDetailsModalProps) => {
  if (!genome) return null;

  const { genome: g, fitness } = genome;

  return (
    <Modal
      title={`🧬 Геном: ${g.id.slice(-8)}`}
      open={open}
      onCancel={onClose}
      width={700}
      footer={[
        <Button key="close" onClick={onClose}>
          Закрыть
        </Button>,
        <Button key="export" type="primary" icon={<DownloadOutlined />} onClick={() => onExport(genome)}>
          Экспорт JSON
        </Button>,
      ]}
    >
      <Descriptions column={2} size="small" bordered>
        <Descriptions.Item label="Алгоритм">{g.algorithm}</Descriptions.Item>
        <Descriptions.Item label="Плечо">x{g.leverage}</Descriptions.Item>
        <Descriptions.Item label="Депозит">${g.depositAmount}</Descriptions.Item>
        <Descriptions.Item label="Поколение">{g.generation}</Descriptions.Item>
        <Descriptions.Item label="Score" span={1}>
          <Text strong style={{ color: '#52c41a' }}>{fitness.score.toFixed(4)}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="Pareto" span={1}>
          {genome.paretoOptimal
            ? <Text strong style={{ color: '#faad14' }}>★ Оптимальный</Text>
            : <Text type="secondary">—</Text>
          }
        </Descriptions.Item>
      </Descriptions>

      <Card title="📊 Результаты" size="small" style={{ marginTop: 16 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="PnL"
              value={fitness.totalPnl}
              precision={2}
              prefix="$"
              valueStyle={{ color: fitness.totalPnl >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Col>
          <Col span={6}>
            <Statistic title="Win Rate" value={fitness.winRate} precision={1} suffix="%" />
          </Col>
          <Col span={6}>
            <Statistic
              title="Max DD"
              value={fitness.maxDrawdown}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#cf1322' }}
            />
          </Col>
          <Col span={6}>
            <Statistic title="Сделок" value={fitness.totalDeals} />
          </Col>
        </Row>
        {fitness.backtestIds && fitness.backtestIds.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <Text type="secondary">Бэктесты: </Text>
            {fitness.backtestIds.map((id) => (
              <a
                key={id}
                href={buildCabinetUrl(`backtests/${id}`)}
                target="_blank"
                rel="noopener noreferrer"
                style={{ marginRight: 8 }}
              >
                #{id}
              </a>
            ))}
          </div>
        )}
      </Card>

      <Card title="📌 Условия входа" size="small" style={{ marginTop: 16 }}>
        <Alert
          type="info"
          message="Условия входа наследуются от базового бота (не оптимизируются)"
          style={{ marginBottom: 8 }}
        />
        {g.entryConditions.length === 0 ? (
          <Text type="secondary">Нет условий</Text>
        ) : (
          <Space wrap>
            {g.entryConditions.map((c, i) => (
              <Tag key={i} color="default">
                {c.indicator} {c.operation} {c.value ?? ''} ({c.interval})
              </Tag>
            ))}
          </Space>
        )}
      </Card>

      <Card title="📈 Сетка DCA (оптимизируется)" size="small" style={{ marginTop: 16 }}>
        <Text type="secondary">Базовый ордер: </Text>
        <Tag color="blue">отступ {roundForDisplay(g.baseOrder.indent)}%, объём {roundForDisplay(g.baseOrder.volume)}%</Tag>
        <br />
        <Text type="secondary">DCA ордера ({g.dcaOrders.length}): </Text>
        {g.dcaOrders.map((o, i) => (
          <Tag key={i} color="blue">{roundForDisplay(o.indent)}% / {roundForDisplay(o.volume)}%</Tag>
        ))}
        <div style={{ marginTop: 8 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>
            Условия в ордерах наследуются от базового бота
          </Text>
        </div>
      </Card>

      <Card title="🎯 Take Profit (оптимизируется)" size="small" style={{ marginTop: 16 }}>
        <Tag color="green">
          Мин. P&L: {roundForDisplay(g.takeProfit.value)}%
        </Tag>
        <div style={{ marginTop: 8 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>
            Условие TP (индикатор) наследуется от базового бота
          </Text>
        </div>
      </Card>

      {g.stopLoss && (
        <Card title="🛑 Stop Loss (оптимизируется)" size="small" style={{ marginTop: 16 }}>
          <Tag color="red">
            Отступ: {roundForDisplay(g.stopLoss.indent)}%
          </Tag>
          {g.stopLoss.termination && (
            <Tag color="volcano" style={{ marginLeft: 8 }}>
              Терминация
            </Tag>
          )}
          {g.stopLoss.conditionalIndent && (
            <Tag color="orange" style={{ marginLeft: 8 }}>
              Условный: {g.stopLoss.conditionalIndent}%
            </Tag>
          )}
          {g.stopLoss.conditions.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">Условия SL: </Text>
              {g.stopLoss.conditions.map((c, i) => (
                <Tag key={i} color="default" style={{ marginTop: 4 }}>
                  {c.indicator} {c.operation} {c.value ?? ''} ({c.interval})
                </Tag>
              ))}
            </div>
          )}
        </Card>
      )}

      <Card title="💰 Депозит (оптимизируется)" size="small" style={{ marginTop: 16 }}>
        <Space>
          <Tag color="purple">Сумма: ${g.depositAmount}</Tag>
          <Tag color="purple">Плечо: x{g.leverage}</Tag>
        </Space>
      </Card>
    </Modal>
  );
};

// ═══════════════════════════════════════════════════════════════════
// ГЛАВНЫЙ КОМПОНЕНТ
// ═══════════════════════════════════════════════════════════════════

const OptimizerPage = ({ extensionReady }: OptimizerPageProps) => {
  // Состояние выбора бота
  const [bots, setBots] = useState<TradingBot[]>([]);
  const [botsLoading, setBotsLoading] = useState(false);
  const [selectedBotId, setSelectedBotId] = useState<number | null>(null);
  const [selectedBot, setSelectedBot] = useState<TradingBot | null>(null);

  // Конфигурация
  const [periodFrom, setPeriodFrom] = useState('');
  const [periodTo, setPeriodTo] = useState('');
  const [symbols, setSymbols] = useState('BTC, ETH, SOL');
  const [geneticConfig, setGeneticConfig] = useState<GeneticConfig>(DEFAULT_GENETIC_CONFIG);
  const [scope, setScope] = useState<OptimizationScope>(DEFAULT_SCOPE);
  const [target, setTarget] = useState<OptimizationTarget>(DEFAULT_TARGET);

  // Состояние оптимизации
  const [status, setStatus] = useState<OptimizationStatus>('idle');
  const [progress, setProgress] = useState<OptimizationProgress>({
    status: 'idle',
    currentGeneration: 0,
    totalGenerations: 0,
    evaluatedGenomes: 0,
    totalBacktests: 0,
    completedBacktests: 0,
    startedAt: null,
    estimatedEndAt: null,
    error: null,
  });
  const [topGenomes, setTopGenomes] = useState<EvaluatedGenome[]>([]);
  const [logs, setLogs] = useState<OptimizationLogEntry[]>([]);

  // Модальное окно деталей генома
  const [selectedGenome, setSelectedGenome] = useState<EvaluatedGenome | null>(null);
  const [genomeModalOpen, setGenomeModalOpen] = useState(false);

  // Модальное окно настроек ордеров
  const [orderSettingsOpen, setOrderSettingsOpen] = useState(false);
  const [orderConfigs, setOrderConfigs] = useState<OrderOptimizationConfig[]>([]);

  // Preview-геном из выбранного бота (для модального окна настроек ордеров)
  const botGenomePreview = useMemo<import('../types/optimizer').BotGenome | null>(() => {
    if (!selectedBot) return null;
    const baseOrderDto = selectedBot.settings?.baseOrder;
    const baseOrder = baseOrderDto
      ? dtoToGridOrderGene(baseOrderDto)
      : { indent: 0, volume: 10, conditions: [] };
    const dcaOrders = (selectedBot.settings?.orders ?? []).map(dtoToGridOrderGene);
    return {
      id: 'preview',
      generation: 0,
      algorithm: (selectedBot.algorithm as 'LONG' | 'SHORT') ?? 'LONG',
      leverage: selectedBot.deposit?.leverage ?? 10,
      depositAmount: selectedBot.deposit?.amount ?? 10,
      entryConditions: [],
      baseOrder,
      dcaOrders,
      takeProfit: { type: 'PERCENT', value: 1, indicator: null },
      stopLoss: null,
      pullUp: null,
      portion: null,
    };
  }, [selectedBot]);

  // Сохранённое состояние для восстановления
  const [savedStateInfo, setSavedStateInfo] = useState<ReturnType<typeof getSavedOptimizerInfo>>(null);

  // Ссылка на оптимизатор
  const optimizerRef = useRef<GeneticOptimizer | null>(null);

  // Проверка сохранённого состояния при загрузке
  useEffect(() => {
    const info = getSavedOptimizerInfo();
    setSavedStateInfo(info);

    // Восстанавливаем топ геномов из сохранённого состояния
    const savedTop = getSavedTopGenomes();
    if (savedTop.length > 0) {
      setTopGenomes(savedTop);
    }
  }, []);

  // Загрузка ботов
  useEffect(() => {
    if (!extensionReady) return;

    setBotsLoading(true);
    fetchBots({ page: 0, size: 100 })
      .then((response) => {
        setBots(response.content);
      })
      .catch((err) => {
        console.error('Failed to load bots:', err);
      })
      .finally(() => {
        setBotsLoading(false);
      });
  }, [extensionReady]);

  // Установка периода по умолчанию (3 месяца)
  useEffect(() => {
    const now = new Date();
    const to = now.toISOString().slice(0, 10);
    const from = new Date(now.setMonth(now.getMonth() - 3)).toISOString().slice(0, 10);
    setPeriodFrom(from);
    setPeriodTo(to);
  }, []);

  // Выбор бота
  const handleBotSelect = useCallback(
    async (botId: number) => {
      setSelectedBotId(botId);
      const bot = bots.find((b) => b.id === botId) ?? null;
      setSelectedBot(bot);
      // Сбрасываем per-order конфиги при смене бота
      setOrderConfigs([]);

      if (bot) {
        // Устанавливаем символы бота
        if (bot.symbols.length > 0) {
          setSymbols(bot.symbols.map((s) => s.split('/')[0]).join(', '));
        }
      }
    },
    [bots],
  );

  // Добавление лога
  const addLog = useCallback((level: OptimizationLogEntry['level'], msg: string) => {
    const entry: OptimizationLogEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: Date.now(),
      level,
      message: msg,
    };
    setLogs((prev) => [...prev, entry]);
  }, []);

  // Расчёт сложности пространства поиска на основе scope
  const searchComplexity = useMemo(() => {
    let complexity = 1;
    let factors: string[] = [];

    if (scope.entryConditionValues) {
      complexity += 2;
      factors.push('пороги индикаторов');
    }
    if (scope.entryConditionIndicators) {
      complexity += 5; // Большой вклад - много вариантов индикаторов
      factors.push('индикаторы входа');
    }
    if (scope.dcaIndents) {
      complexity += 2;
      factors.push('отступы DCA');
    }
    if (scope.dcaVolumes) {
      complexity += 2;
      factors.push('объёмы DCA');
    }
    if (scope.dcaConditions) {
      complexity += 3;
      factors.push('условия DCA');
    }
    if (scope.dcaStructure) {
      complexity += 4;
      factors.push('структура сетки');
    }
    if (scope.takeProfit) {
      complexity += 1;
      factors.push('тейк-профит');
    }
    if (scope.takeProfitIndicator) {
      complexity += 3;
      factors.push('индикатор TP');
    }
    if (scope.stopLoss) {
      complexity += 2;
      factors.push('стоп-лосс');
    }
    if (scope.leverage) {
      complexity += 1;
      factors.push('плечо');
    }

    // Уровень сложности
    let level: 'low' | 'medium' | 'high' | 'extreme';
    let color: string;
    let recommendedPopulation: number;
    let recommendedGenerations: number;

    if (complexity <= 3) {
      level = 'low';
      color = '#52c41a';
      recommendedPopulation = 15;
      recommendedGenerations = 8;
    } else if (complexity <= 8) {
      level = 'medium';
      color = '#1890ff';
      recommendedPopulation = 20;
      recommendedGenerations = 12;
    } else if (complexity <= 15) {
      level = 'high';
      color = '#faad14';
      recommendedPopulation = 30;
      recommendedGenerations = 15;
    } else {
      level = 'extreme';
      color = '#ff4d4f';
      recommendedPopulation = 40;
      recommendedGenerations = 20;
    }

    return { complexity, factors, level, color, recommendedPopulation, recommendedGenerations };
  }, [scope]);

  // Расчёт оценки
  const estimatedBacktests = useMemo(() => {
    const symbolCount = parseSymbols(symbols).length;
    return geneticConfig.populationSize * geneticConfig.generations * symbolCount;
  }, [geneticConfig.populationSize, geneticConfig.generations, symbols]);

  const estimatedTime = useMemo(() => {
    // Поллинг ~15 сек в среднем + пользовательская задержка
    const avgPollMs = 15_000;
    const delayMs = geneticConfig.backtestDelaySeconds * 1000;
    const totalMs = estimatedBacktests * (avgPollMs + delayMs);
    const hours = Math.floor(totalMs / 3600000);
    const minutes = Math.ceil((totalMs % 3600000) / 60000);
    return hours > 0 ? `${hours}ч ${minutes}мин` : `${minutes}мин`;
  }, [estimatedBacktests, geneticConfig.backtestDelaySeconds]);

  // Запуск оптимизации
  const handleStart = useCallback(async () => {
    if (!selectedBot || selectedBotId === null) {
      message.error('Выберите бота для оптимизации');
      return;
    }

    // Валидация
    const symbolsList = parseSymbols(symbols);
    if (symbolsList.length === 0) {
      message.error('Укажите хотя бы одну монету');
      return;
    }

    if (!periodFrom || !periodTo) {
      message.error('Укажите период тестирования');
      return;
    }

    // Сброс состояния
    setStatus('running');
    setLogs([]);
    setTopGenomes([]);

    // Создаём конфигурацию
    const scopeWithOrders: OptimizationScope = {
      ...scope,
      orderConfigs: orderConfigs.length > 0 ? orderConfigs : undefined,
    };
    const config: OptimizationRunConfig = createOptimizerConfig({
      botId: selectedBotId,
      symbols: symbolsList,
      periodFrom: `${periodFrom}T00:00:00.000Z`,
      periodTo: `${periodTo}T23:59:59.999Z`,
      genetic: geneticConfig,
      scope: scopeWithOrders,
      target,
    });

    // Callbacks
    const callbacks: OptimizerCallbacks = {
      onLog: addLog,
      onProgress: (p) => setProgress(p),
      onGenomeEvaluated: (genome) => {
        setTopGenomes((prev) => {
          const updated = [...prev];
          const existingIdx = updated.findIndex((g) => g.genome.id === genome.genome.id);
          if (existingIdx >= 0) {
            updated[existingIdx] = genome;
          } else {
            updated.push(genome);
          }
          updated.sort((a, b) => b.fitness.score - a.fitness.score);
          return updated.slice(0, 10);
        });
      },
      onGenerationComplete: (gen, top) => {
        setTopGenomes(top);
      },
    };

    // Создаём и запускаем оптимизатор
    const optimizer = new GeneticOptimizer(config, callbacks);
    optimizerRef.current = optimizer;

    setProgress({
      status: 'running',
      currentGeneration: 0,
      totalGenerations: geneticConfig.generations,
      evaluatedGenomes: 0,
      totalBacktests: estimatedBacktests,
      completedBacktests: 0,
      startedAt: Date.now(),
      estimatedEndAt: Date.now() + estimatedBacktests * 31000,
      error: null,
    });

    try {
      const results = await optimizer.start();
      setStatus('completed');
      setTopGenomes(results);
      message.success(`Оптимизация завершена! Лучший score: ${results[0]?.fitness.score.toFixed(3) ?? 'N/A'}`);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Неизвестная ошибка';
      setStatus('error');
      setProgress((prev) => ({ ...prev, status: 'error', error: errorMsg }));
      message.error(`Ошибка: ${errorMsg}`);
    } finally {
      optimizerRef.current = null;
    }
  }, [selectedBot, selectedBotId, symbols, periodFrom, periodTo, geneticConfig, scope, target, orderConfigs, estimatedBacktests, addLog]);

  // Пауза
  const handlePause = useCallback(() => {
    if (optimizerRef.current) {
      optimizerRef.current.pause();
      setStatus('paused');
    }
  }, []);

  // Продолжить (из паузы)
  const handleResume = useCallback(() => {
    if (optimizerRef.current) {
      optimizerRef.current.unpause();
      setStatus('running');
    }
  }, []);

  // Стоп
  const handleStop = useCallback(() => {
    if (optimizerRef.current) {
      optimizerRef.current.stop();
    }
    setStatus('idle');
    optimizerRef.current = null;
    // Обновляем информацию о сохранённом состоянии
    setSavedStateInfo(getSavedOptimizerInfo());
  }, []);

  // Восстановление из сохранённого состояния
  const handleRestoreFromSaved = useCallback(async () => {
    if (!hasSavedOptimizerState()) {
      message.error('Нет сохранённого состояния для восстановления');
      return;
    }

    const callbacks: OptimizerCallbacks = {
      onLog: addLog,
      onProgress: (p) => setProgress(p),
      onGenomeEvaluated: (genome) => {
        setTopGenomes((prev) => {
          const updated = [...prev];
          const existingIdx = updated.findIndex((g) => g.genome.id === genome.genome.id);
          if (existingIdx >= 0) {
            updated[existingIdx] = genome;
          } else {
            updated.push(genome);
          }
          updated.sort((a, b) => b.fitness.score - a.fitness.score);
          return updated.slice(0, 10);
        });
      },
      onGenerationComplete: (_, top) => setTopGenomes([...top]),
    };

    const optimizer = GeneticOptimizer.fromSavedState(callbacks);
    if (!optimizer) {
      message.error('Не удалось восстановить состояние');
      return;
    }

    // Сразу показываем сохранённый топ геномов
    const savedTop = optimizer.getAllTimeTop();
    if (savedTop.length > 0) {
      setTopGenomes(savedTop);
    }

    optimizerRef.current = optimizer;
    setStatus('running');
    setSavedStateInfo(null);
    addLog('info', '🔄 Восстановление оптимизации из сохранённого состояния...');

    try {
      const results = await optimizer.resume();
      setTopGenomes(results);
      setStatus('idle');
      addLog('success', `✅ Восстановленная оптимизация завершена. Найдено ${results.length} лучших геномов.`);
      message.success('Оптимизация успешно завершена!');
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Неизвестная ошибка';
      addLog('error', `❌ Ошибка: ${msg}`);
      setStatus('idle');
      // Обновляем информацию о сохранённом состоянии (оно может быть обновлено)
      setSavedStateInfo(getSavedOptimizerInfo());
      message.error(`Ошибка: ${msg}. Прогресс сохранён, можно продолжить позже.`);
    }

    optimizerRef.current = null;
  }, [addLog]);

  // Очистка сохранённого состояния
  const handleClearSavedState = useCallback(() => {
    GeneticOptimizer.clearSavedState();
    setSavedStateInfo(null);
    message.info('Сохранённое состояние очищено');
  }, []);

  // Выбор генома для просмотра
  const handleGenomeSelect = useCallback((genome: EvaluatedGenome) => {
    setSelectedGenome(genome);
    setGenomeModalOpen(true);
  }, []);

  // Экспорт генома в JSON
  const handleExportGenome = useCallback((genome: EvaluatedGenome) => {
    const strategy = genomeToStrategy(genome.genome, {
      exchange: selectedBot?.exchange ?? 'BINANCE_FUTURES',
      symbol: symbols.split(/[,\s]+/)[0] ?? 'BTC/USDT',
      quoteCurrency: 'USDT',
    });

    const exportData = {
      genome: genome.genome,
      fitness: genome.fitness,
      strategy,
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `genome-${genome.genome.id.slice(-8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    message.success('Геном экспортирован');
  }, [selectedBot, symbols]);

  // Копировать JSON стратегии в буфер
  const handleCopyStrategy = useCallback((genome: EvaluatedGenome) => {
    const strategy = genomeToStrategy(genome.genome, {
      exchange: selectedBot?.exchange ?? 'BINANCE_FUTURES',
      symbol: symbols.split(/[,\s]+/)[0] ?? 'BTC/USDT',
      quoteCurrency: 'USDT',
    });

    navigator.clipboard.writeText(JSON.stringify(strategy, null, 2))
      .then(() => message.success('Стратегия скопирована в буфер'))
      .catch(() => message.error('Не удалось скопировать'));
  }, [selectedBot, symbols]);

  const progressPercent = useMemo(() => {
    if (progress.totalBacktests === 0) return 0;
    return Math.round((progress.completedBacktests / progress.totalBacktests) * 100);
  }, [progress.completedBacktests, progress.totalBacktests]);

  // ═══════════════════════════════════════════════════════════════════
  // РЕНДЕР
  // ═══════════════════════════════════════════════════════════════════

  if (!extensionReady) {
    return (
      <div className="page-container">
        <Alert
          type="warning"
          message="Расширение недоступно"
          description="Для работы оптимизатора необходимо открыть страницу через расширение Veles Tools"
        />
      </div>
    );
  }

  return (
    <div className="page-container">
      <PageHeader
        title="🧬 AI Оптимизатор стратегий"
        description="Генетический поиск лучших параметров бота"
      />

      {/* Уведомление о сохранённом состоянии */}
      {savedStateInfo && status === 'idle' && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="Найдено незавершённое задание"
          description={
            <Space direction="vertical" size="small">
              <Text>
                Бот ID: {savedStateInfo.botId} • Поколение {savedStateInfo.generation}/{savedStateInfo.totalGenerations} • 
                Оценено: {savedStateInfo.evaluatedGenomes} геномов
              </Text>
              <Text type="secondary">
                Сохранено: {savedStateInfo.savedAt.toLocaleString()}
              </Text>
              <Space>
                <Button type="primary" size="small" icon={<PlayCircleOutlined />} onClick={handleRestoreFromSaved}>
                  Продолжить
                </Button>
                <Button size="small" danger onClick={handleClearSavedState}>
                  Удалить
                </Button>
              </Space>
            </Space>
          }
        />
      )}

      <Row gutter={[16, 16]}>
        {/* Левая колонка - настройки */}
        <Col xs={24} lg={10}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* Выбор бота */}
            <Card title="📌 Базовый бот" size="small">
              <Select
                style={{ width: '100%' }}
                placeholder="Выберите бота для оптимизации"
                loading={botsLoading}
                value={selectedBotId}
                onChange={handleBotSelect}
                options={bots.map((bot) => ({
                  value: bot.id,
                  label: `${bot.name} (${bot.algorithm}, ${bot.exchange})`,
                }))}
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
                disabled={status === 'running'}
              />
              {selectedBot && (
                <div style={{ marginTop: 12 }}>
                  <Text type="secondary">
                    {selectedBot.symbols.join(', ')} • {selectedBot.algorithm} •{' '}
                    Депозит: {selectedBot.deposit.amount} {selectedBot.deposit.currency}
                  </Text>
                </div>
              )}
            </Card>

            {/* Период */}
            <Card title="📅 Период тестирования" size="small">
              <Space>
                <Input
                  type="date"
                  value={periodFrom}
                  onChange={(e) => setPeriodFrom(e.target.value)}
                  style={{ width: 150 }}
                  disabled={status === 'running'}
                />
                <Text>—</Text>
                <Input
                  type="date"
                  value={periodTo}
                  onChange={(e) => setPeriodTo(e.target.value)}
                  style={{ width: 150 }}
                  disabled={status === 'running'}
                />
              </Space>
            </Card>

            {/* Монеты */}
            <Card title="🪙 Монеты для тестирования" size="small">
              <TextArea
                rows={2}
                value={symbols}
                onChange={(e) => setSymbols(e.target.value)}
                placeholder="BTC, ETH, SOL, DOGE"
                disabled={status === 'running'}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                Разделитель: запятая или пробел. Каждая монета = отдельный бэктест.
              </Text>
            </Card>

            {/* Область оптимизации */}
            <Card title="⚙️ Что оптимизируем" size="small">
              <Space direction="vertical">
                <Checkbox
                  checked={scope.entryConditionValues}
                  onChange={(e) => setScope({ ...scope, entryConditionValues: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Пороговые значения индикаторов входа
                </Checkbox>
                <Checkbox
                  checked={scope.entryConditionIndicators}
                  onChange={(e) => setScope({ ...scope, entryConditionIndicators: e.target.checked })}
                  disabled={status === 'running'}
                >
                  <Text>Сами индикаторы входа</Text>
                  <Tag color="orange" style={{ marginLeft: 8 }}>
                    много вариантов
                  </Tag>
                </Checkbox>
                <Checkbox
                  checked={scope.dcaIndents}
                  onChange={(e) => setScope({ ...scope, dcaIndents: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Отступы сетки DCA
                </Checkbox>
                <Checkbox
                  checked={scope.dcaVolumes}
                  onChange={(e) => setScope({ ...scope, dcaVolumes: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Объёмы сетки DCA
                </Checkbox>
                <Checkbox
                  checked={scope.dcaConditions}
                  onChange={(e) => setScope({ ...scope, dcaConditions: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Условия в ордерах DCA
                </Checkbox>
                <Checkbox
                  checked={scope.dcaStructure}
                  onChange={(e) => setScope({ ...scope, dcaStructure: e.target.checked })}
                  disabled={status === 'running'}
                >
                  <Text>Структура сетки (кол-во ордеров)</Text>
                  <Tag color="red" style={{ marginLeft: 8 }}>
                    эксперимент
                  </Tag>
                </Checkbox>
                <Checkbox
                  checked={scope.takeProfit}
                  onChange={(e) => setScope({ ...scope, takeProfit: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Значение тейк-профита
                </Checkbox>
                <Checkbox
                  checked={scope.takeProfitIndicator}
                  onChange={(e) => setScope({ ...scope, takeProfitIndicator: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Индикатор тейк-профита
                </Checkbox>
                <Checkbox
                  checked={scope.stopLoss}
                  onChange={(e) => setScope({ ...scope, stopLoss: e.target.checked })}
                  disabled={status === 'running'}
                >
                  <Text>Стоп-лосс (если есть в боте)</Text>
                  <Tooltip title="Оптимизирует отступ стоп-лосса от 1% до 50%. Работает только если стоп-лосс включён в базовом боте.">
                    <Text type="secondary" style={{ marginLeft: 4, fontSize: 11 }}>ⓘ</Text>
                  </Tooltip>
                </Checkbox>
                <Checkbox
                  checked={scope.leverage}
                  onChange={(e) => setScope({ ...scope, leverage: e.target.checked })}
                  disabled={status === 'running'}
                >
                  Плечо
                </Checkbox>

                {/* Кнопка настройки per-order оптимизации */}
                {selectedBot && (
                  <Button
                    icon={<SettingOutlined />}
                    onClick={() => setOrderSettingsOpen(true)}
                    disabled={status === 'running'}
                    block
                    style={{ marginTop: 8 }}
                  >
                    Настройки по ордерам ({orderConfigs.filter((c) => c.locked).length} заф.)
                  </Button>
                )}
              </Space>
            </Card>

            {/* Генетические настройки */}
            <Card title="🧬 Настройки алгоритма" size="small">
              <Row gutter={[16, 12]}>
                <Col span={12}>
                  <Text type="secondary">Размер популяции</Text>
                  <InputNumber
                    min={5}
                    max={50}
                    value={geneticConfig.populationSize}
                    onChange={(v) => setGeneticConfig({ ...geneticConfig, populationSize: v ?? 20 })}
                    style={{ width: '100%' }}
                    disabled={status === 'running'}
                  />
                </Col>
                <Col span={12}>
                  <Text type="secondary">Поколений</Text>
                  <InputNumber
                    min={3}
                    max={50}
                    value={geneticConfig.generations}
                    onChange={(v) => setGeneticConfig({ ...geneticConfig, generations: v ?? 10 })}
                    style={{ width: '100%' }}
                    disabled={status === 'running'}
                  />
                </Col>
                <Col span={12}>
                  <Text type="secondary">Мутация %</Text>
                  <Slider
                    min={10}
                    max={50}
                    value={geneticConfig.mutationRate * 100}
                    onChange={(v) => setGeneticConfig({ ...geneticConfig, mutationRate: v / 100 })}
                    disabled={status === 'running'}
                  />
                </Col>
                <Col span={12}>
                  <Text type="secondary">Скрещивание %</Text>
                  <Slider
                    min={50}
                    max={90}
                    value={geneticConfig.crossoverRate * 100}
                    onChange={(v) => setGeneticConfig({ ...geneticConfig, crossoverRate: v / 100 })}
                    disabled={status === 'running'}
                  />
                </Col>
                <Col span={24}>
                  <Text type="secondary">
                    Задержка между бэктестами: <Text strong>{geneticConfig.backtestDelaySeconds} сек</Text>
                  </Text>
                  <Slider
                    min={3}
                    max={60}
                    value={geneticConfig.backtestDelaySeconds}
                    onChange={(v) => setGeneticConfig({ ...geneticConfig, backtestDelaySeconds: v })}
                    disabled={status === 'running'}
                    marks={{
                      3: '3с',
                      15: '15с',
                      31: '31с',
                      60: '60с',
                    }}
                    tooltip={{ formatter: (v) => `${v} сек` }}
                  />
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    ⚠️ Меньше 31 сек — риск получить 429 (rate limit). Рекомендуется 31+ сек.
                  </Text>
                </Col>
              </Row>
            </Card>

            {/* Целевая метрика */}
            <Card title="🎯 Целевая метрика" size="small">
              <Select
                style={{ width: '100%' }}
                value={target.metric}
                onChange={(v) => setTarget({ ...target, metric: v })}
                options={[
                  { value: 'pnlToRisk', label: 'Прибыль / Риск (PnL / MaxDrawdown)' },
                  { value: 'pnl', label: 'Чистая прибыль (PnL)' },
                  { value: 'pnlPerDay', label: 'Прибыль в день' },
                  { value: 'winRate', label: 'Win Rate %' },
                  { value: 'composite', label: 'Составная метрика (настроить веса)' },
                ]}
                disabled={status === 'running'}
              />
            </Card>

            {/* Оценка и сложность */}
            <Card
              size="small"
              title={
                <Space>
                  📊 Оценка
                  <Tag color={searchComplexity.color}>
                    Сложность: {searchComplexity.complexity}
                  </Tag>
                </Space>
              }
            >
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="Бэктестов"
                    value={estimatedBacktests}
                  />
                </Col>
                <Col span={8}>
                  <Statistic title="Время" value={estimatedTime} />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Сложность"
                    value={searchComplexity.level === 'low' ? 'Низкая' : 
                           searchComplexity.level === 'medium' ? 'Средняя' :
                           searchComplexity.level === 'high' ? 'Высокая' : 'Экстрим'}
                    valueStyle={{ color: searchComplexity.color }}
                  />
                </Col>
              </Row>

              {searchComplexity.factors.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <Text type="secondary">Оптимизируем: </Text>
                  {searchComplexity.factors.map((f, i) => (
                    <Tag key={i} style={{ marginBottom: 4 }}>{f}</Tag>
                  ))}
                </div>
              )}

              {(geneticConfig.populationSize < searchComplexity.recommendedPopulation ||
                geneticConfig.generations < searchComplexity.recommendedGenerations) && (
                <Alert
                  type="warning"
                  message={
                    <Space direction="vertical" size={0}>
                      <Text>Рекомендуется для данной сложности:</Text>
                      <Text type="secondary">
                        Популяция: {searchComplexity.recommendedPopulation}, 
                        Поколений: {searchComplexity.recommendedGenerations}
                      </Text>
                    </Space>
                  }
                  style={{ marginTop: 12 }}
                  showIcon
                />
              )}

              <Alert
                type="info"
                message={`Оптимизатор ждёт результат каждого бэктеста + ${geneticConfig.backtestDelaySeconds} сек задержка`}
                style={{ marginTop: 12 }}
                showIcon
              />
            </Card>

            {/* Кнопки управления */}
            <Flex gap={8}>
              {status === 'idle' && (
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleStart}
                  disabled={!selectedBot}
                  block
                  size="large"
                >
                  🚀 Запустить оптимизацию
                </Button>
              )}
              {status === 'running' && (
                <>
                  <Button icon={<PauseCircleOutlined />} onClick={handlePause} size="large">
                    Пауза
                  </Button>
                  <Button danger icon={<StopOutlined />} onClick={handleStop} size="large">
                    Стоп
                  </Button>
                </>
              )}
              {status === 'paused' && (
                <>
                  <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleResume} size="large">
                    Продолжить
                  </Button>
                  <Button danger icon={<StopOutlined />} onClick={handleStop} size="large">
                    Стоп
                  </Button>
                </>
              )}
              {(status === 'completed' || status === 'error') && (
                <Button type="primary" icon={<ExperimentOutlined />} onClick={handleStart} block size="large">
                  Запустить заново
                </Button>
              )}
            </Flex>
          </Space>
        </Col>

        {/* Правая колонка - результаты */}
        <Col xs={24} lg={14}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* Прогресс */}
            {status !== 'idle' && (
              <Card title="📊 Прогресс" size="small">
                <Progress
                  percent={progressPercent}
                  status={status === 'running' ? 'active' : status === 'error' ? 'exception' : 'normal'}
                />
                <Row gutter={16} style={{ marginTop: 12 }}>
                  <Col span={8}>
                    <Statistic
                      title="Поколение"
                      value={progress.currentGeneration}
                      suffix={`/ ${progress.totalGenerations}`}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Бэктестов"
                      value={progress.completedBacktests}
                      suffix={`/ ${progress.totalBacktests}`}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic title="Оценено геномов" value={progress.evaluatedGenomes} />
                  </Col>
                </Row>
                {progress.error && (
                  <Alert type="error" message={progress.error} style={{ marginTop: 12 }} />
                )}
              </Card>
            )}

            {/* Лучший геном */}
            {topGenomes.length > 0 && (
              <Card
                title="🏆 Лучшая комбинация"
                size="small"
                extra={
                  <Space>
                    <Tag color="gold">Score: {topGenomes[0].fitness.score.toFixed(3)}</Tag>
                    <Tooltip title="Копировать JSON стратегии">
                      <Button
                        icon={<CopyOutlined />}
                        size="small"
                        onClick={() => handleCopyStrategy(topGenomes[0])}
                      />
                    </Tooltip>
                    <Tooltip title="Скачать геном">
                      <Button
                        icon={<DownloadOutlined />}
                        size="small"
                        onClick={() => handleExportGenome(topGenomes[0])}
                      />
                    </Tooltip>
                  </Space>
                }
              >
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title="PnL"
                      value={topGenomes[0].fitness.totalPnl}
                      precision={2}
                      prefix="$"
                      valueStyle={{ color: topGenomes[0].fitness.totalPnl >= 0 ? '#3f8600' : '#cf1322' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="Win Rate"
                      value={topGenomes[0].fitness.winRate}
                      precision={1}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="Max DD"
                      value={topGenomes[0].fitness.maxDrawdown}
                      precision={1}
                      suffix="%"
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="Сделок"
                      value={topGenomes[0].fitness.totalDeals}
                    />
                  </Col>
                </Row>
              </Card>
            )}

            {/* Топ геномов */}
            {topGenomes.length > 0 && (
              <Card
                title="📈 Топ-10 комбинаций"
                size="small"
                extra={<Text type="secondary">Кликните для деталей</Text>}
              >
                <TopGenomesTable genomes={topGenomes} onSelect={handleGenomeSelect} />
              </Card>
            )}

            {/* Логи */}
            <Card title="📝 Лог выполнения" size="small">
              <LogViewer logs={logs} />
            </Card>

            {/* Каталог индикаторов */}
            <IndicatorCatalogView />
          </Space>
        </Col>
      </Row>

      {/* Модальное окно деталей генома */}
      <GenomeDetailsModal
        genome={selectedGenome}
        open={genomeModalOpen}
        onClose={() => setGenomeModalOpen(false)}
        onExport={handleExportGenome}
      />

      {/* Модальное окно настроек ордеров */}
      <OrderSettingsModal
        open={orderSettingsOpen}
        onClose={() => setOrderSettingsOpen(false)}
        genome={botGenomePreview}
        orderConfigs={orderConfigs}
        onSave={setOrderConfigs}
      />
    </div>
  );
};

export default OptimizerPage;
