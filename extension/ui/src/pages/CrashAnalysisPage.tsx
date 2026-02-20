/**
 * Страница анализа ликвидационных проливов BTC
 * 
 * Позволяет:
 * - Загрузить исторические данные с Binance
 * - Провести feature engineering
 * - Выполнить статистический анализ
 * - Найти правила предсказания crash'ей
 * - Провести walk-forward валидацию
 */

import {
  AlertOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  LoadingOutlined,
  PlayCircleOutlined,
  QuestionCircleOutlined,
  RobotOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Divider,
  InputNumber,
  Progress,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useCallback, useEffect, useState } from 'react';
import { runCrashAnalysis } from '../services/crashAnalysis';
import {
  DEFAULT_ML_CONFIG,
  prepareMLDataset,
  runMLAnalysis,
} from '../services/crashML';
import type {
  MLAnalysisResult,
  MLConfig,
  MLModelResult,
} from '../services/crashML';
import type {
  AnalysisPeriod,
  AnalysisTimeframe,
  CombinedRule,
  CrashAnalysisConfig,
  CrashAnalysisResult,
  DataLoadProgress,
  FeatureBar,
  FeatureCorrelation,
  FeatureImportance,
  WalkForwardResult,
} from '../types/crashAnalysis';
import { BARS_PER_24H, DEFAULT_CRASH_ANALYSIS_CONFIG } from '../types/crashAnalysis';

const { Title, Text, Paragraph } = Typography;

// ═══════════════════════════════════════════════════════════════════
// КОМПОНЕНТЫ ТАБЛИЦ
// ═══════════════════════════════════════════════════════════════════

const CorrelationsTable = ({ correlations }: { correlations: FeatureCorrelation[] }) => {
  const columns: ColumnsType<FeatureCorrelation> = [
    {
      title: 'Признак',
      dataIndex: 'featureName',
      key: 'featureName',
      render: (name: string) => <Text code>{name}</Text>,
    },
    {
      title: 'Корреляция',
      dataIndex: 'correlation',
      key: 'correlation',
      render: (val: number) => (
        <Tag color={val > 0 ? 'red' : 'green'}>
          {val > 0 ? '+' : ''}{val.toFixed(4)}
        </Tag>
      ),
      sorter: (a, b) => Math.abs(b.correlation) - Math.abs(a.correlation),
    },
    {
      title: 'p-value',
      dataIndex: 'pValue',
      key: 'pValue',
      render: (val: number) => val.toExponential(2),
    },
    {
      title: 'Значимость',
      dataIndex: 'isSignificant',
      key: 'isSignificant',
      render: (sig: boolean) => (
        sig ? <Tag color="success">✓ Значимо</Tag> : <Tag>Не значимо</Tag>
      ),
    },
  ];

  return (
    <Table
      dataSource={correlations}
      columns={columns}
      rowKey="featureName"
      size="small"
      pagination={false}
    />
  );
};

const FeatureImportanceTable = ({ importance }: { importance: FeatureImportance[] }) => {
  const columns: ColumnsType<FeatureImportance> = [
    {
      title: '#',
      dataIndex: 'rank',
      key: 'rank',
      width: 50,
    },
    {
      title: 'Признак',
      dataIndex: 'featureName',
      key: 'featureName',
      render: (name: string) => <Text code>{name}</Text>,
    },
    {
      title: 'Важность',
      dataIndex: 'importance',
      key: 'importance',
      render: (val: number) => (
        <Progress
          percent={Math.round(val * 100)}
          size="small"
          format={(p) => `${(val * 100).toFixed(1)}%`}
        />
      ),
    },
  ];

  return (
    <Table
      dataSource={importance}
      columns={columns}
      rowKey="featureName"
      size="small"
      pagination={false}
    />
  );
};

const RulesTable = ({ rules, crashThreshold }: { rules: CombinedRule[]; crashThreshold: number }) => {
  const columns: ColumnsType<CombinedRule> = [
    {
      title: 'Условия',
      key: 'conditions',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          {record.conditions.map((c, i) => (
            <Text key={i} code>
              {c.featureName} {c.operator} {c.threshold.toFixed(2)}
            </Text>
          ))}
        </Space>
      ),
    },
    {
      title: 'P(crash)',
      key: 'crashProbability',
      render: (_, record) => (
        <Tag color={record.crashProbability > 0.3 ? 'red' : record.crashProbability > 0.15 ? 'orange' : 'default'}>
          {(record.crashProbability * 100).toFixed(1)}%
        </Tag>
      ),
      sorter: (a, b) => b.crashProbability - a.crashProbability,
    },
    {
      title: 'Lift',
      dataIndex: 'lift',
      key: 'lift',
      render: (val: number) => (
        <Tag color={val > 3 ? 'volcano' : val > 2 ? 'orange' : 'default'}>
          {val.toFixed(2)}x
        </Tag>
      ),
      sorter: (a, b) => b.lift - a.lift,
    },
    {
      title: 'Срабатываний',
      dataIndex: 'support',
      key: 'support',
    },
    {
      title: 'Crash\'ей',
      dataIndex: 'crashes',
      key: 'crashes',
    },
  ];

  return (
    <Table
      dataSource={rules}
      columns={columns}
      rowKey={(_, idx) => `rule-${idx}`}
      size="small"
      pagination={false}
    />
  );
};

const WalkForwardTable = ({ results }: { results: WalkForwardResult[] }) => {
  const columns: ColumnsType<WalkForwardResult> = [
    {
      title: 'Train период',
      key: 'trainPeriod',
      render: (_, r) => `${r.trainPeriod.from} — ${r.trainPeriod.to}`,
    },
    {
      title: 'Test период',
      key: 'testPeriod',
      render: (_, r) => `${r.testPeriod.from} — ${r.testPeriod.to}`,
    },
    {
      title: 'Train crash rate',
      dataIndex: 'trainCrashRate',
      key: 'trainCrashRate',
      render: (val: number) => `${(val * 100).toFixed(2)}%`,
    },
    {
      title: 'Test crash rate',
      dataIndex: 'testCrashRate',
      key: 'testCrashRate',
      render: (val: number) => `${(val * 100).toFixed(2)}%`,
    },
    {
      title: 'Precision',
      dataIndex: 'testPrecision',
      key: 'testPrecision',
      render: (val: number) => (
        <Tag color={val > 0.3 ? 'green' : val > 0.15 ? 'orange' : 'default'}>
          {(val * 100).toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: 'Recall',
      dataIndex: 'testRecall',
      key: 'testRecall',
      render: (val: number) => `${(val * 100).toFixed(1)}%`,
    },
  ];

  return (
    <Table
      dataSource={results}
      columns={columns}
      rowKey={(_, idx) => `wf-${idx}`}
      size="small"
      pagination={false}
    />
  );
};

// ═══════════════════════════════════════════════════════════════════
// ГЛАВНЫЙ КОМПОНЕНТ
// ═══════════════════════════════════════════════════════════════════

const CrashAnalysisPage = () => {
  // Конфигурация
  const [config, setConfig] = useState<CrashAnalysisConfig>(DEFAULT_CRASH_ANALYSIS_CONFIG);
  
  // Состояние
  const [progress, setProgress] = useState<DataLoadProgress>({ 
    stage: 'idle', 
    progress: 0, 
    message: '' 
  });
  const [result, setResult] = useState<CrashAnalysisResult | null>(null);
  const [mlResult, setMlResult] = useState<MLAnalysisResult | null>(null);
  const [featuresData, setFeaturesData] = useState<FeatureBar[] | null>(null);
  const [enableML, setEnableML] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // ML Config
  const [mlConfig] = useState<MLConfig>(DEFAULT_ML_CONFIG);

  // Проверка состояния выполнения
  const isRunning = [
    'loading-ohlcv', 'loading-oi', 'loading-funding', 
    'loading-feargreed', 'loading-spot',
    'processing', 'analyzing', 'ml-training'
  ].includes(progress.stage);

  // Таймер прогресса
  useEffect(() => {
    if (!startTime || !isRunning) return;
    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime, isRunning]);

  // Форматирование времени
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return m > 0 ? `${m}м ${s}с` : `${s}с`;
  };

  // Расчёт оставшегося времени
  const getETA = () => {
    if (progress.progress <= 0 || elapsedTime <= 0) return null;
    const totalEstimate = (elapsedTime / progress.progress) * 100;
    const remaining = Math.max(0, Math.round(totalEstimate - elapsedTime));
    return remaining;
  };

  // Запуск анализа
  const handleStartAnalysis = useCallback(async () => {
    setError(null);
    setResult(null);
    setMlResult(null);
    setFeaturesData(null);
    setStartTime(Date.now());
    setElapsedTime(0);
    
    try {
      const analysisResult = await runCrashAnalysis(config, {
        onProgress: setProgress,
      });
      setResult(analysisResult);
      
      // Если включён ML, запускаем ML анализ
      if (enableML && analysisResult.features) {
        setFeaturesData(analysisResult.features);
        
        setProgress({ stage: 'ml-training', progress: 0, message: 'Запуск ML анализа...' });
        
        // Подготовка датасета
        const featureColumns = [
          'oiZscore24h', 'fundingZscore24h', 'atrZscore72h',
          'fearGreedZscore7d', 'basisZscore24h',
          'takerDeltaRatio', 'volumeZscore24h',
          'priceChangePct1h', 'priceChangePct4h', 'priceChangePct24h',
        ].filter(col => analysisResult.features?.some(f => f[col as keyof FeatureBar] !== null));
        
        const dataset = prepareMLDataset(
          analysisResult.features as unknown as Array<Record<string, number | null>>,
          featureColumns,
          'crashNext6h'
        );
        
        if (dataset.X.length < 1000) {
          console.warn('Недостаточно данных для ML анализа');
        } else {
          const mlAnalysisResult = await runMLAnalysis(
            dataset,
            mlConfig,
            (stage, pct, msg) => {
              setProgress({ stage: 'ml-training', progress: pct, message: msg });
            }
          );
          setMlResult(mlAnalysisResult);
        }
        
        setProgress({ stage: 'done', progress: 100, message: 'Анализ завершён!' });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Неизвестная ошибка';
      setError(message);
      setProgress({ stage: 'error', progress: 0, message: '', error: message });
    }
  }, [config, enableML, mlConfig]);

  // Иконка статуса
  const getStageIcon = () => {
    switch (progress.stage) {
      case 'idle': return <ExperimentOutlined />;
      case 'loading-ohlcv':
      case 'loading-oi':
      case 'loading-funding':
      case 'loading-feargreed':
      case 'loading-spot':
      case 'processing':
      case 'analyzing':
        return <LoadingOutlined spin />;
      case 'ml-training':
        return <RobotOutlined spin style={{ color: '#1890ff' }} />;
      case 'done': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error': return <WarningOutlined style={{ color: '#ff4d4f' }} />;
      default: return <ExperimentOutlined />;
    }
  };

  // Экспорт результатов для ML
  const handleExportForML = useCallback(() => {
    if (!result) return;
    
    // Формируем информацию о конфигурации и результатах
    const exportData = {
      config: {
        crashThresholdPct: config.crashThresholdPct,
        crashWindowBars: config.crashWindowBars,
        minRuleSupport: config.minRuleSupport,
      },
      datasetInfo: result.datasetInfo,
      correlations: result.correlations,
      topRules: result.topRules.map(rule => ({
        conditions: rule.conditions.map(c => `${c.featureName} ${c.operator} ${c.threshold.toFixed(4)}`),
        probability: rule.crashProbability,
        support: rule.support,
        lift: rule.lift,
      })),
      walkForwardResults: result.walkForwardResults.map(wf => ({
        trainPeriod: wf.trainPeriod,
        testPeriod: wf.testPeriod,
        precision: wf.testPrecision,
        recall: wf.testRecall,
      })),
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `crash_analysis_results_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, config]);

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Заголовок */}
        <div>
          <Title level={2}>
            <AlertOutlined /> Анализ ликвидационных проливов BTC
          </Title>
          <Paragraph type="secondary">
            Поиск статистически значимых признаков, предшествующих резким падениям цены.
            Использует исторические данные Binance Futures, Fear & Greed Index, и Spot-Futures Basis.
          </Paragraph>
        </div>

        {/* Информация о данных */}
        <Row gutter={16}>
          <Col span={12}>
            <Alert
              type="info"
              showIcon
              icon={<QuestionCircleOutlined />}
              message="Доступные данные"
              description={
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  <li><Text type="success">✓</Text> OHLCV, Volume — 5+ лет</li>
                  <li><Text type="success">✓</Text> Funding Rate — 3+ года</li>
                  <li><Text type="success">✓</Text> Fear & Greed Index — с 2018</li>
                  <li><Text type="success">✓</Text> Spot-Futures Basis — 5+ лет</li>
                  <li><Text type="warning">~</Text> Open Interest — ~30 дней</li>
                </ul>
              }
            />
          </Col>
          <Col span={12}>
            <Alert
              type="info"
              showIcon
              icon={<RobotOutlined />}
              message="ML классификация в браузере"
              description={
                <span>
                  Logistic Regression, Decision Tree, Random Forest — всё работает прямо здесь.
                  Включите переключатель ML ниже и нажмите «Запустить анализ».
                </span>
              }
            />
          </Col>
        </Row>

        {/* Настройки */}
        <Card title="⚙️ Параметры анализа" size="small">
          {/* Первый ряд: Таймфрейм и Период */}
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                <Text type="secondary">Таймфрейм</Text>
                <Select
                  value={config.timeframe}
                  onChange={(v) => {
                    const bars24h = BARS_PER_24H[v as AnalysisTimeframe];
                    setConfig({ 
                      ...config, 
                      timeframe: v as AnalysisTimeframe,
                      zscore24hBars: bars24h,
                      zscore72hBars: bars24h * 3,
                      crashWindowBars: bars24h / 2, // 12 часов
                    });
                  }}
                  disabled={isRunning}
                  style={{ width: '100%' }}
                  options={[
                    { value: '15m', label: '15 минут (детально, медленно)' },
                    { value: '1h', label: '1 час (оптимально)' },
                    { value: '4h', label: '4 часа (быстро)' },
                  ]}
                />
              </Space>
            </Col>
            <Col span={6}>
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                <Text type="secondary">Период данных</Text>
                <Select
                  value={config.periodYears}
                  onChange={(v) => setConfig({ ...config, periodYears: v as AnalysisPeriod })}
                  disabled={isRunning}
                  style={{ width: '100%' }}
                  options={[
                    { value: 1, label: '1 год (быстро)' },
                    { value: 2, label: '2 года (оптимально)' },
                    { value: 3, label: '3 года' },
                    { value: 5, label: '5 лет (много данных)' },
                  ]}
                />
              </Space>
            </Col>
            <Col span={6}>
              <Space direction="vertical" size={4}>
                <Text type="secondary">Порог падения (%)</Text>
                <InputNumber
                  value={config.crashThresholdPct}
                  onChange={(v) => v && setConfig({ ...config, crashThresholdPct: v })}
                  min={3}
                  max={20}
                  step={1}
                  disabled={isRunning}
                  style={{ width: '100%' }}
                />
              </Space>
            </Col>
            <Col span={6}>
              <Space direction="vertical" size={4}>
                <Text type="secondary">
                  Окно прогноза (баров)
                  <Tooltip title={`${config.crashWindowBars} баров = ${config.crashWindowBars * (config.timeframe === '15m' ? 0.25 : config.timeframe === '1h' ? 1 : 4)} часов`}>
                    <QuestionCircleOutlined style={{ marginLeft: 4 }} />
                  </Tooltip>
                </Text>
                <InputNumber
                  value={config.crashWindowBars}
                  onChange={(v) => v && setConfig({ ...config, crashWindowBars: v })}
                  min={4}
                  max={96}
                  step={4}
                  disabled={isRunning}
                  style={{ width: '100%' }}
                />
              </Space>
            </Col>
          </Row>
          
          {/* Второй ряд: остальные настройки */}
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col span={6}>
              <Space direction="vertical" size={4}>
                <Text type="secondary">Мин. срабатываний правила</Text>
                <InputNumber
                  value={config.minRuleSupport}
                  onChange={(v) => v && setConfig({ ...config, minRuleSupport: v })}
                  min={5}
                  max={100}
                  disabled={isRunning}
                  style={{ width: '100%' }}
                />
              </Space>
            </Col>
            <Col span={6}>
              <Space direction="vertical" size={4}>
                <Text type="secondary">ML классификация</Text>
                <Switch
                  checked={enableML}
                  onChange={setEnableML}
                  disabled={isRunning}
                  checkedChildren={<RobotOutlined />}
                  unCheckedChildren="Off"
                />
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {enableML ? 'Logistic Reg, Decision Tree, Random Forest' : 'Только правила'}
                </Text>
              </Space>
            </Col>
            <Col span={12}>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleStartAnalysis}
                loading={isRunning}
                size="large"
                style={{ width: '100%', height: 64 }}
              >
                {isRunning ? 'Анализ...' : 'Запустить анализ'}
              </Button>
            </Col>
          </Row>
          
          {/* Предупреждение о тяжёлых настройках */}
          {(config.timeframe === '15m' || config.periodYears >= 3) && (
            <Alert
              type="warning"
              showIcon
              style={{ marginTop: 16 }}
              message="Внимание: большой объём данных"
              description={
                <span>
                  Текущие настройки загрузят много данных (~{Math.round(
                    config.periodYears * 365 * 24 * (config.timeframe === '15m' ? 4 : config.timeframe === '1h' ? 1 : 0.25)
                  ).toLocaleString()} баров). 
                  Рекомендуем использовать таймфрейм 1h и период 1-2 года для комфортной работы.
                </span>
              }
            />
          )}
        </Card>

        {/* Прогресс */}
        {progress.stage !== 'idle' && (
          <Card size="small">
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  {getStageIcon()}
                  <Text strong>{progress.message}</Text>
                </Space>
              </Col>
              <Col>
                {isRunning && (
                  <Space size="large">
                    <Text type="secondary">
                      ⏱️ Прошло: <Text strong>{formatTime(elapsedTime)}</Text>
                    </Text>
                    {getETA() !== null && getETA()! > 0 && (
                      <Text type="secondary">
                        ⏳ Осталось: ~<Text strong>{formatTime(getETA()!)}</Text>
                      </Text>
                    )}
                  </Space>
                )}
                {progress.stage === 'done' && (
                  <Tag color="success">Готово за {formatTime(elapsedTime)}</Tag>
                )}
              </Col>
            </Row>
            {isRunning && (
              <Progress 
                percent={Math.round(progress.progress)} 
                status="active"
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
                style={{ marginTop: 12 }} 
              />
            )}
          </Card>
        )}

        {/* Ошибка */}
        {error && (
          <Alert type="error" message="Ошибка" description={error} showIcon />
        )}

        {/* Результаты */}
        {result && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {/* Общая статистика */}
            <Card 
              title="📊 Датасет" 
              size="small"
              extra={
                <Button
                  icon={<DownloadOutlined />}
                  onClick={handleExportForML}
                  size="small"
                >
                  Экспорт для ML
                </Button>
              }
            >
              <Row gutter={16}>
                <Col span={4}>
                  <Statistic title="Всего баров" value={result.datasetInfo.totalBars} />
                </Col>
                <Col span={4}>
                  <Statistic title="Crash'ей" value={result.datasetInfo.barsWithCrash} />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="Base crash rate"
                    value={(result.datasetInfo.baseCrashRate * 100).toFixed(2)}
                    suffix="%"
                  />
                </Col>
                <Col span={6}>
                  <Statistic title="Период" value={`${result.datasetInfo.periodFrom} — ${result.datasetInfo.periodTo}`} />
                </Col>
                <Col span={6}>
                  <Statistic title="Таймфрейм" value={result.datasetInfo.timeframe} />
                </Col>
              </Row>
            </Card>

            {/* Рекомендации */}
            {result.recommendations.length > 0 && (
              <Alert
                type="success"
                icon={<CheckCircleOutlined />}
                message="Ключевые выводы"
                description={
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {result.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                }
              />
            )}

            {/* Детальные результаты */}
            <Collapse
              defaultActiveKey={['rules']}
              items={[
                {
                  key: 'correlations',
                  label: (
                    <Space>
                      <BarChartOutlined />
                      Корреляции признаков с crash_next_6h
                    </Space>
                  ),
                  children: <CorrelationsTable correlations={result.correlations} />,
                },
                {
                  key: 'importance',
                  label: (
                    <Space>
                      <BarChartOutlined />
                      Feature Importance
                    </Space>
                  ),
                  children: <FeatureImportanceTable importance={result.featureImportance} />,
                },
                {
                  key: 'rules',
                  label: (
                    <Space>
                      <AlertOutlined />
                      Найденные правила (top 20)
                    </Space>
                  ),
                  children: (
                    <>
                      <Alert
                        type="info"
                        message={`Правила для предсказания падения ≥${config.crashThresholdPct}% в течение ${config.crashWindowBars} баров (${config.crashWindowBars * 15 / 60} часов)`}
                        style={{ marginBottom: 16 }}
                      />
                      <RulesTable rules={result.topRules} crashThreshold={config.crashThresholdPct} />
                    </>
                  ),
                },
                {
                  key: 'walkforward',
                  label: (
                    <Space>
                      <ExperimentOutlined />
                      Walk-Forward валидация
                    </Space>
                  ),
                  children: (
                    <>
                      <Alert
                        type="warning"
                        message="Валидация показывает как правила, найденные на train периоде, работают на новых данных (test)"
                        style={{ marginBottom: 16 }}
                      />
                      <WalkForwardTable results={result.walkForwardResults} />
                    </>
                  ),
                },
              ]}
            />

            {/* Итоговые правила в читаемом формате */}
            {result.topRules.length > 0 && (
              <Card
                title="🎯 Лучшее правило"
                extra={
                  <Button icon={<DownloadOutlined />} size="small" disabled>
                    Экспорт (скоро)
                  </Button>
                }
              >
                <Descriptions column={1} bordered size="small">
                  <Descriptions.Item label="Условия">
                    <Space direction="vertical">
                      {result.topRules[0].conditions.map((c, i) => (
                        <Tag key={i} color="blue">
                          {c.featureName} {c.operator} {c.threshold.toFixed(2)}
                        </Tag>
                      ))}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="Вероятность crash'а">
                    <Tag color="red" style={{ fontSize: 16 }}>
                      {(result.topRules[0].crashProbability * 100).toFixed(1)}%
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="Lift (превышение базовой вероятности)">
                    <Tag color="volcano" style={{ fontSize: 16 }}>
                      {result.topRules[0].lift.toFixed(2)}x
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="Статистика">
                    Условия выполнялись {result.topRules[0].support} раз, 
                    из них crash произошёл {result.topRules[0].crashes} раз
                  </Descriptions.Item>
                </Descriptions>

                <Alert
                  type="info"
                  style={{ marginTop: 16 }}
                  message="Интерпретация"
                  description={
                    <>
                      <Paragraph>
                        Когда одновременно выполняются условия:
                      </Paragraph>
                      <ul>
                        {result.topRules[0].conditions.map((c, i) => (
                          <li key={i}>
                            <Text code>{c.featureName}</Text> {c.operator} {c.threshold.toFixed(2)}
                          </li>
                        ))}
                      </ul>
                      <Paragraph>
                        Вероятность падения цены BTC на {config.crashThresholdPct}% или более 
                        в течение следующих {config.crashWindowBars * 15 / 60} часов составляет{' '}
                        <Text strong>{(result.topRules[0].crashProbability * 100).toFixed(1)}%</Text>,
                        что в {result.topRules[0].lift.toFixed(1)} раз выше базовой вероятности{' '}
                        ({(result.datasetInfo.baseCrashRate * 100).toFixed(2)}%).
                      </Paragraph>
                    </>
                  }
                />
              </Card>
            )}

            {/* ML Results */}
            {mlResult && (
              <Card title={<><RobotOutlined /> ML Классификация</>} size="small">
                <Tabs
                  items={[
                    {
                      key: 'models',
                      label: '📊 Модели',
                      children: (
                        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                          <Row gutter={16}>
                            {mlResult.models.map((model) => (
                              <Col span={8} key={model.name}>
                                <Card 
                                  size="small" 
                                  title={model.name}
                                  style={{ 
                                    borderColor: model.metrics.rocAuc >= 0.6 ? '#52c41a' : 
                                                 model.metrics.rocAuc >= 0.55 ? '#faad14' : '#d9d9d9'
                                  }}
                                >
                                  <Row gutter={[8, 8]}>
                                    <Col span={12}>
                                      <Statistic 
                                        title="AUC" 
                                        value={model.metrics.rocAuc} 
                                        precision={3}
                                        valueStyle={{ 
                                          color: model.metrics.rocAuc >= 0.6 ? '#3f8600' : 
                                                 model.metrics.rocAuc >= 0.55 ? '#cf1322' : undefined,
                                          fontSize: 18
                                        }}
                                      />
                                    </Col>
                                    <Col span={12}>
                                      <Statistic 
                                        title="Precision" 
                                        value={model.metrics.precision} 
                                        precision={3}
                                        valueStyle={{ fontSize: 18 }}
                                      />
                                    </Col>
                                    <Col span={12}>
                                      <Statistic 
                                        title="Recall" 
                                        value={model.metrics.recall} 
                                        precision={3}
                                        valueStyle={{ fontSize: 18 }}
                                      />
                                    </Col>
                                    <Col span={12}>
                                      <Statistic 
                                        title="F1" 
                                        value={model.metrics.f1Score} 
                                        precision={3}
                                        valueStyle={{ fontSize: 18 }}
                                      />
                                    </Col>
                                  </Row>
                                  <Divider style={{ margin: '8px 0' }} />
                                  <Text type="secondary" style={{ fontSize: 11 }}>
                                    TP: {model.metrics.confusionMatrix.tp}, 
                                    FP: {model.metrics.confusionMatrix.fp}, 
                                    TN: {model.metrics.confusionMatrix.tn}, 
                                    FN: {model.metrics.confusionMatrix.fn}
                                  </Text>
                                </Card>
                              </Col>
                            ))}
                          </Row>
                          <Alert
                            type={mlResult.bestModel.metrics.rocAuc >= 0.6 ? 'success' : 
                                  mlResult.bestModel.metrics.rocAuc >= 0.55 ? 'warning' : 'info'}
                            message={`Лучшая модель: ${mlResult.bestModel.name} (AUC: ${mlResult.bestModel.metrics.rocAuc.toFixed(3)})`}
                            description={
                              mlResult.bestModel.metrics.rocAuc >= 0.6 
                                ? 'Модель показывает хорошую предсказательную способность'
                                : mlResult.bestModel.metrics.rocAuc >= 0.55
                                  ? 'Модель показывает слабый сигнал, требуется осторожность'
                                  : 'Модель близка к случайному угадыванию (AUC ~0.5)'
                            }
                            showIcon
                          />
                        </Space>
                      ),
                    },
                    {
                      key: 'features',
                      label: '📈 Feature Importance',
                      children: (
                        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                          <Table
                            dataSource={mlResult.featureImportance.slice(0, 15).map((f, i) => ({ 
                              key: i, 
                              rank: i + 1,
                              ...f 
                            }))}
                            columns={[
                              { title: '#', dataIndex: 'rank', width: 40 },
                              { title: 'Признак', dataIndex: 'feature', ellipsis: true },
                              { 
                                title: 'Важность', 
                                dataIndex: 'importance', 
                                render: (v: number) => (
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                    <div 
                                      style={{ 
                                        width: `${v * 100}%`, 
                                        height: 12, 
                                        backgroundColor: '#1677ff',
                                        borderRadius: 2,
                                        minWidth: 4
                                      }} 
                                    />
                                    <Text type="secondary">{(v * 100).toFixed(1)}%</Text>
                                  </div>
                                ),
                                width: 200
                              },
                            ]}
                            pagination={false}
                            size="small"
                          />
                          <Text type="secondary">
                            Важность признаков агрегирована по всем моделям (нормализованная сумма)
                          </Text>
                        </Space>
                      ),
                    },
                    {
                      key: 'rules',
                      label: '📋 Извлечённые правила',
                      children: (
                        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                          {mlResult.extractedRules.slice(0, 5).map((rule, idx) => (
                            <Card 
                              key={idx} 
                              size="small"
                              title={<Tag color="blue">Правило #{idx + 1}</Tag>}
                              extra={
                                <Space>
                                  <Tag color="red">P(crash): {(rule.probability * 100).toFixed(1)}%</Tag>
                                  <Tag color="volcano">Lift: {rule.lift.toFixed(2)}x</Tag>
                                  <Tag>Support: {rule.support}</Tag>
                                </Space>
                              }
                            >
                              <Space wrap>
                                {rule.conditions.map((cond, i) => (
                                  <Tag key={i} color="geekblue">
                                    {cond.feature} {cond.operator} {cond.threshold.toFixed(2)}
                                  </Tag>
                                ))}
                              </Space>
                            </Card>
                          ))}
                          {mlResult.extractedRules.length === 0 && (
                            <Alert 
                              type="info" 
                              message="Не удалось извлечь значимые правила из Decision Tree" 
                            />
                          )}
                        </Space>
                      ),
                    },
                    {
                      key: 'validation',
                      label: '🔄 Walk-Forward',
                      children: (
                        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                          <Alert
                            type="info"
                            message="Walk-Forward валидация"
                            description={`Модели обучались на ${mlResult.validationInfo.trainSize} примерах и тестировались на ${mlResult.validationInfo.testSize} примерах. Использовалось ${mlResult.validationInfo.nSplits} последовательных сплитов с шагом ${mlResult.validationInfo.stepSize} баров.`}
                          />
                          <Row gutter={16}>
                            <Col span={6}>
                              <Statistic 
                                title="Размер Train" 
                                value={mlResult.validationInfo.trainSize} 
                                suffix="баров"
                              />
                            </Col>
                            <Col span={6}>
                              <Statistic 
                                title="Размер Test" 
                                value={mlResult.validationInfo.testSize} 
                                suffix="баров"
                              />
                            </Col>
                            <Col span={6}>
                              <Statistic 
                                title="Сплитов" 
                                value={mlResult.validationInfo.nSplits} 
                              />
                            </Col>
                            <Col span={6}>
                              <Statistic 
                                title="Всего примеров" 
                                value={mlResult.datasetInfo.totalSamples} 
                              />
                            </Col>
                          </Row>
                          <Divider />
                          <Row gutter={16}>
                            <Col span={8}>
                              <Statistic 
                                title="Crash примеров" 
                                value={mlResult.datasetInfo.crashSamples} 
                                valueStyle={{ color: '#cf1322' }}
                              />
                            </Col>
                            <Col span={8}>
                              <Statistic 
                                title="Non-crash примеров" 
                                value={mlResult.datasetInfo.nonCrashSamples} 
                                valueStyle={{ color: '#3f8600' }}
                              />
                            </Col>
                            <Col span={8}>
                              <Statistic 
                                title="Базовый crash rate" 
                                value={(mlResult.datasetInfo.crashRate * 100).toFixed(2)} 
                                suffix="%"
                              />
                            </Col>
                          </Row>
                        </Space>
                      ),
                    },
                  ]}
                />
              </Card>
            )}
          </Space>
        )}
      </Space>
    </div>
  );
};

export default CrashAnalysisPage;
