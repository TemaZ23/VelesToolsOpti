/**
 * Конвертер между геномом и конфигурацией бота Veles
 * Преобразует генетическое представление в реальный payload для API
 */

import type { BotStrategy } from '../api/backtestRunner';
import type {
  BotDepositConfigDto,
  BotOrderDto,
  BotProfitConfigDto,
  BotSettingsDto,
  BotStopLossConfigDto,
  StrategyConditionDto,
} from '../api/bots.dtos';
import type { BotGenome, ConditionGene, GridOrderGene, StopLossGene, TakeProfitGene } from '../types/optimizer';

/**
 * Алиас для BotStrategy (теперь содержит все нужные поля).
 * Оставлен для обратной совместимости.
 */
export type FullBotStrategy = BotStrategy;

// ═══════════════════════════════════════════════════════════════════
// МАППИНГ ИНДИКАТОРОВ
// ═══════════════════════════════════════════════════════════════════

/**
 * Маппинг ID индикаторов на типы в API Veles
 * TODO: Уточнить реальные названия типов в API
 */
const INDICATOR_TYPE_MAP: Record<string, string> = {
  ADX: 'ADX',
  EMA: 'EMA',
  SMA: 'SMA',
  MACD: 'MACD',
  PSAR: 'PARABOLIC_SAR',
  BB: 'BOLLINGER_BANDS',
  KELTNER: 'KELTNER_CHANNEL',
  TURTLE: 'TURTLE_ZONE',
  DONCHIAN: 'DONCHIAN_CHANNEL',
  RSI: 'RSI',
  RSI_LEVELS: 'RSI_LEVELS',
  CCI: 'CCI',
  CCI_LEVELS: 'CCI_LEVELS',
  CMO: 'CMO',
  WILLIAMS_R: 'WILLIAMS_R',
  STOCHASTIC: 'STOCHASTIC',
  STOCHASTIC_LEVELS: 'STOCHASTIC_LEVELS',
  MFI: 'MFI',
  ROC: 'ROC',
  MOMENTUM: 'MOMENTUM',
  ATR: 'ATR',
  ATR_PERCENT: 'ATR_PERCENT',
  VOLUME: 'VOLUME',
  OBV: 'OBV',
  VWAP: 'VWAP',
};

/**
 * Маппинг интервалов (внутреннее представление -> API)
 */
const INTERVAL_MAP: Record<string, string> = {
  '1m': 'ONE_MINUTE',
  '5m': 'FIVE_MINUTES',
  '15m': 'FIFTEEN_MINUTES',
  '30m': 'THIRTY_MINUTES',
  '1h': 'ONE_HOUR',
  '4h': 'FOUR_HOURS',
  '1d': 'ONE_DAY',
  // Также поддержка прямых значений API (если уже в правильном формате)
  'ONE_MINUTE': 'ONE_MINUTE',
  'FIVE_MINUTES': 'FIVE_MINUTES',
  'FIFTEEN_MINUTES': 'FIFTEEN_MINUTES',
  'THIRTY_MINUTES': 'THIRTY_MINUTES',
  'ONE_HOUR': 'ONE_HOUR',
  'FOUR_HOURS': 'FOUR_HOURS',
  'ONE_DAY': 'ONE_DAY',
};

/**
 * Округление числа до разумной точности для API
 */
const roundForApi = (value: number, decimals = 2): number => {
  return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
};

/**
 * Маппинг операций
 */
const OPERATION_MAP: Record<string, string> = {
  '>': 'GREATER',
  '<': 'LESS',
  '>=': 'GREATER_OR_EQUAL',
  '<=': 'LESS_OR_EQUAL',
  '==': 'EQUAL',
  CROSS_UP: 'CROSS_UP',
  CROSS_DOWN: 'CROSS_DOWN',
};

// ═══════════════════════════════════════════════════════════════════
// КОНВЕРТАЦИЯ ГЕНОВ В DTO
// ═══════════════════════════════════════════════════════════════════

/**
 * Конвертировать ConditionGene в StrategyConditionDto
 */
export const conditionGeneToDto = (gene: ConditionGene): StrategyConditionDto => {
  const indicatorType = INDICATOR_TYPE_MAP[gene.indicator] ?? gene.indicator;
  
  return {
    type: 'INDICATOR', // Всегда INDICATOR для условий
    indicator: indicatorType,
    interval: INTERVAL_MAP[gene.interval] ?? gene.interval,
    basic: gene.basic, // Важно! Сохраняем флаг basic
    value: gene.value !== null ? roundForApi(gene.value, 4) : null,
    operation: gene.basic ? null : (OPERATION_MAP[gene.operation] ?? gene.operation), // basic индикаторы не имеют operation
    closed: gene.closed,
    reverse: gene.reverse,
  };
};

/**
 * Конвертировать GridOrderGene в BotOrderDto
 */
export const gridOrderGeneToDto = (gene: GridOrderGene): BotOrderDto => {
  return {
    indent: roundForApi(gene.indent, 2),
    volume: roundForApi(gene.volume, 2),
    conditions: gene.conditions.length > 0 ? gene.conditions.map(conditionGeneToDto) : null,
  };
};

/**
 * Конвертировать TakeProfitGene в BotProfitConfigDto
 */
export const takeProfitGeneToDto = (gene: TakeProfitGene, quoteCurrency: string): BotProfitConfigDto => {
  return {
    type: gene.type,
    currency: quoteCurrency,
    checkPnl: roundForApi(gene.value, 2),
    conditions: gene.indicator ? [conditionGeneToDto(gene.indicator)] : null,
  };
};

/**
 * Конвертировать StopLossGene в BotStopLossConfigDto
 */
export const stopLossGeneToDto = (gene: StopLossGene): BotStopLossConfigDto => {
  return {
    indent: roundForApi(gene.indent, 2),
    termination: gene.termination,
    conditionalIndent: gene.conditionalIndent !== null ? roundForApi(gene.conditionalIndent, 2) : null,
    conditionalIndentType: gene.conditionalIndent !== null ? 'PERCENT' : null,
    conditions: gene.conditions.length > 0 ? gene.conditions.map(conditionGeneToDto) : null,
  };
};

// ═══════════════════════════════════════════════════════════════════
// КОНВЕРТАЦИЯ ГЕНОМА В СТРАТЕГИЮ БОТА
// ═══════════════════════════════════════════════════════════════════

export interface GenomeToStrategyOptions {
  exchange: string;
  symbol: string;
  quoteCurrency: string;
  apiKeyId?: number;
}

/**
 * Конвертировать BotGenome в FullBotStrategy (для создания бэктеста)
 */
export const genomeToStrategy = (genome: BotGenome, options: GenomeToStrategyOptions): FullBotStrategy => {
  const { exchange, symbol, quoteCurrency } = options;

  // Разбираем символ на base/quote
  const [baseCurrency, quote] = symbol.includes('/') ? symbol.split('/') : [symbol, quoteCurrency];

  // Условия входа
  const conditions = genome.entryConditions.map(conditionGeneToDto);

  // Настройки сетки
  const settings: BotSettingsDto = {
    type: 'GRID',
    includePosition: true,
    indentType: 'PERCENT',
    baseOrder: gridOrderGeneToDto(genome.baseOrder),
    orders: genome.dcaOrders.map(gridOrderGeneToDto),
  };

  // Депозит
  const deposit: BotDepositConfigDto = {
    amount: roundForApi(genome.depositAmount, 2),
    leverage: Math.round(genome.leverage),
    marginType: 'CROSS',
    currency: quoteCurrency,
  };

  // Профит
  const profit = takeProfitGeneToDto(genome.takeProfit, quoteCurrency);

  // Стоп-лосс
  const stopLoss = genome.stopLoss ? stopLossGeneToDto(genome.stopLoss) : null;

  const strategy: FullBotStrategy = {
    id: null,
    name: `Optimizer_Gen${genome.generation}_${genome.id.slice(-6)}`,
    symbol,
    symbols: [symbol],
    pair: {
      exchange,
      type: 'FUTURES',
      from: baseCurrency,
      to: quote,
      symbol: `${baseCurrency}${quote}`,
    },
    exchange,
    algorithm: genome.algorithm,
    status: 'FINISHED',
    settings: settings as FullBotStrategy['settings'],
    conditions,
    profit,
    deposit,
    stopLoss,
    pullUp: genome.pullUp !== null ? roundForApi(genome.pullUp, 2) : null,
    portion: genome.portion !== null ? roundForApi(genome.portion, 2) : null,
    commissions: null,
    useWicks: true,
    from: null,
    to: null,
    cursor: null,
    includePosition: true,
    public: false,
  };

  return strategy;
};

/**
 * Применить геном к копии базовой стратегии.
 * 
 * Модифицирует параметры в зависимости от scope:
 * - Отступы и объёмы ордеров (indent, volume) - всегда
 * - Депозит и плечо - всегда (из базового бота)
 * - Значение take profit - всегда
 * - Условия входа (entry conditions) - если scope.entryConditions
 * - Условия DCA ордеров - если scope.dcaConditions
 * - Индикатор TP - если scope.takeProfitIndicator
 */
export const applyGenomeToStrategy = (
  baseStrategy: FullBotStrategy,
  genome: BotGenome,
  options: { 
    symbol: string; 
    quoteCurrency: string;
    applyConditions?: boolean; // Применять изменённые conditions (экспериментально)
  },
): FullBotStrategy => {
  // Глубокое клонирование базовой стратегии
  const strategy: FullBotStrategy = JSON.parse(JSON.stringify(baseStrategy));

  const { symbol, quoteCurrency, applyConditions = false } = options;
  const [baseCurrency, quote] = symbol.includes('/') ? symbol.split('/') : [symbol, quoteCurrency];

  // Обновляем символ
  strategy.symbol = symbol;
  strategy.symbols = [symbol];
  strategy.pair = {
    ...strategy.pair,
    from: baseCurrency,
    to: quote,
    symbol: `${baseCurrency}${quote}`,
  };

  // ═══════════════════════════════════════════════════════════════════
  // ПРИМЕНЯЕМ ЧИСЛОВЫЕ ИЗМЕНЕНИЯ ИЗ ГЕНОМА
  // ═══════════════════════════════════════════════════════════════════

  // Депозит (константа из базового бота, не мутируется)
  // leverage тоже константа если scope.leverage = false
  if (strategy.deposit) {
    strategy.deposit.amount = roundForApi(genome.depositAmount, 2);
    strategy.deposit.leverage = genome.leverage;
  }

  // Take profit value
  if (strategy.profit && genome.takeProfit) {
    strategy.profit.checkPnl = roundForApi(genome.takeProfit.value, 2);
    
    // Применяем индикатор TP если включено
    if (applyConditions && genome.takeProfit.indicator) {
      strategy.profit.conditions = [conditionGeneToDto(genome.takeProfit.indicator)];
    }
  }

  // Базовый ордер - indent и volume
  if (strategy.settings?.baseOrder && genome.baseOrder) {
    strategy.settings.baseOrder.indent = roundForApi(genome.baseOrder.indent, 2);
    strategy.settings.baseOrder.volume = roundForApi(genome.baseOrder.volume, 2);
  }

  // DCA ордера - indent, volume и опционально conditions
  if (strategy.settings?.orders && genome.dcaOrders) {
    const minLength = Math.min(strategy.settings.orders.length, genome.dcaOrders.length);
    for (let i = 0; i < minLength; i++) {
      strategy.settings.orders[i].indent = roundForApi(genome.dcaOrders[i].indent, 2);
      strategy.settings.orders[i].volume = roundForApi(genome.dcaOrders[i].volume, 2);
      
      // Применяем conditions DCA если включено
      if (applyConditions && genome.dcaOrders[i].conditions.length > 0) {
        strategy.settings.orders[i].conditions = genome.dcaOrders[i].conditions.map(conditionGeneToDto);
      }
    }
  }

  // Применяем entry conditions если включено
  if (applyConditions && genome.entryConditions.length > 0) {
    strategy.conditions = genome.entryConditions.map(conditionGeneToDto);
  }

  // ═══════════════════════════════════════════════════════════════════
  // ОЧИСТКА ДЛЯ БЭКТЕСТА
  // ═══════════════════════════════════════════════════════════════════

  // Сбрасываем ID для создания нового бэктеста
  strategy.id = null;
  strategy.name = `Optimizer_Gen${genome.generation}_${genome.id.slice(-6)}`;
  strategy.status = 'FINISHED';
  strategy.substatus = undefined;
  strategy.lastFail = undefined;

  // Удаляем apiKey - для бэктеста не нужен
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  delete (strategy as any).apiKey;

  return strategy;
};

// ═══════════════════════════════════════════════════════════════════
// ПАРСИНГ СТРАТЕГИИ БОТА В ГЕНОМ
// ═══════════════════════════════════════════════════════════════════

/**
 * Конвертировать StrategyConditionDto в ConditionGene
 */
export const dtoToConditionGene = (dto: StrategyConditionDto): ConditionGene => {
  // Обратный маппинг типа индикатора
  const indicatorId =
    Object.entries(INDICATOR_TYPE_MAP).find(([_, v]) => v === dto.indicator)?.[0] ?? dto.indicator ?? 'RSI';

  // Обратный маппинг операции
  const operation =
    (Object.entries(OPERATION_MAP).find(([_, v]) => v === dto.operation)?.[0] as ConditionGene['operation']) ?? '>';

  // Сохраняем интервал как есть (в формате API: ONE_HOUR и т.д.)
  // INTERVAL_MAP поддерживает оба формата
  const interval = dto.interval ?? 'ONE_HOUR';

  // basic - важный флаг для индикаторов-сигналов (BB, TURTLE_ZONE, RSI_LEVELS и т.д.)
  const basic = dto.basic === true;

  return {
    indicator: indicatorId,
    interval: interval as ConditionGene['interval'],
    value: dto.value !== null && dto.value !== undefined ? roundForApi(dto.value, 4) : null,
    operation,
    closed: dto.closed ?? false,
    reverse: dto.reverse ?? false,
    basic,
  };
};

/**
 * Конвертировать BotOrderDto в GridOrderGene
 */
export const dtoToGridOrderGene = (dto: BotOrderDto): GridOrderGene => {
  return {
    indent: dto.indent ?? 0,
    volume: dto.volume ?? 10,
    conditions: dto.conditions?.map(dtoToConditionGene) ?? [],
  };
};

/**
 * Конвертировать FullBotStrategy (или BotStrategy) в BotGenome
 */
export const strategyToGenome = (strategy: FullBotStrategy | BotStrategy, generation = 0): BotGenome => {
  const id = `genome-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Приводим к расширенному типу для безопасного доступа к полям
  const fullStrategy = strategy as FullBotStrategy;

  // Парсим условия входа - сначала проверяем settings.conditions (может быть в некоторых версиях API)
  const settingsConditions = fullStrategy.settings?.conditions;
  const rootConditions = fullStrategy.conditions;

  let entryConditions: ConditionGene[] = [];

  // Пытаемся найти условия в разных местах
  if (Array.isArray(settingsConditions)) {
    entryConditions = (settingsConditions as unknown as StrategyConditionDto[]).map(dtoToConditionGene);
  } else if (Array.isArray(rootConditions)) {
    entryConditions = rootConditions.map(dtoToConditionGene);
  }

  // Парсим базовый ордер
  const baseOrderDto = fullStrategy.settings?.baseOrder;
  const baseOrder: GridOrderGene = baseOrderDto
    ? dtoToGridOrderGene(baseOrderDto)
    : { indent: 0, volume: 10, conditions: [] };

  // Парсим DCA ордера
  const dcaOrders: GridOrderGene[] = (fullStrategy.settings?.orders ?? []).map(dtoToGridOrderGene);

  // Парсим тейк-профит
  const profitDto = fullStrategy.profit;
  const takeProfit: TakeProfitGene = {
    type: (profitDto?.type as TakeProfitGene['type']) ?? 'PERCENT',
    value: profitDto?.checkPnl ?? 1,
    indicator: profitDto?.conditions?.[0] ? dtoToConditionGene(profitDto.conditions[0]) : null,
  };

  // Парсим стоп-лосс
  const stopLossDto = fullStrategy.stopLoss;
  const stopLoss: StopLossGene | null = stopLossDto
    ? {
        indent: stopLossDto.indent ?? 5,
        termination: stopLossDto.termination ?? false,
        conditionalIndent: stopLossDto.conditionalIndent ?? null,
        conditions: stopLossDto.conditions?.map(dtoToConditionGene) ?? [],
      }
    : null;

  // Парсим депозит
  const depositDto = fullStrategy.deposit;
  const leverage = depositDto?.leverage ?? 10;
  const depositAmount = depositDto?.amount ?? 10;

  return {
    id,
    generation,
    algorithm: (fullStrategy.algorithm as 'LONG' | 'SHORT') ?? 'LONG',
    leverage,
    depositAmount,
    entryConditions,
    baseOrder,
    dcaOrders,
    takeProfit,
    stopLoss,
    pullUp: fullStrategy.pullUp ?? null,
    portion: fullStrategy.portion ?? null,
  };
};
