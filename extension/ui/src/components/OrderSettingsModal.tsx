/**
 * Модальное окно настроек оптимизации ордеров.
 *
 * Отображает текущие параметры каждого ордера (базовый + DCA) выбранного бота
 * и позволяет:
 *  – зафиксировать ордер (не мутировать indent/volume);
 *  – включить/выключить оптимизацию indent и volume по отдельности;
 *  – задать допустимый диапазон indent и volume для мутации.
 */

import { LockOutlined, UnlockOutlined } from '@ant-design/icons';
import {
  Button,
  Checkbox,
  InputNumber,
  Modal,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { BotGenome, GridOrderGene, OrderOptimizationConfig, TakeProfitOptimizationConfig } from '../types/optimizer';

const { Text } = Typography;

// ═══════════════════════════════════════════════════════════════════
// PROPS
// ═══════════════════════════════════════════════════════════════════

interface OrderSettingsModalProps {
  open: boolean;
  onClose: () => void;
  /** Базовый геном, из которого берутся текущие значения ордеров */
  genome: BotGenome | null;
  /** Текущие per-order конфиги (могут быть пустыми если ещё не настроены) */
  orderConfigs: OrderOptimizationConfig[];
  /** Callback при сохранении настроек */
  onSave: (configs: OrderOptimizationConfig[]) => void;
  /** Показать секцию тейк-профита */
  showTakeProfit: boolean;
  /** Текущая конфигурация TP */
  takeProfitConfig: TakeProfitOptimizationConfig | null;
  /** Callback при сохранении TP-конфига */
  onSaveTakeProfit: (config: TakeProfitOptimizationConfig) => void;
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

/** Собрать дефолтный конфиг для ордера исходя из его текущих значений */
const buildDefaultConfig = (
  index: number,
  order: GridOrderGene,
): OrderOptimizationConfig => {
  const indent = order.indent;
  const volume = order.volume;

  // Диапазон по умолчанию: ±50 % от текущего значения, но не менее 0.01
  const indentLo = Math.max(0, +(indent * 0.5).toFixed(2));
  const indentHi = +(indent * 1.5 || 1).toFixed(2);
  const volumeLo = Math.max(1, +(volume * 0.5).toFixed(2));
  const volumeHi = +(volume * 1.5 || 5).toFixed(2);

  return {
    index,
    locked: false,
    optimizeIndent: true,
    indentRange: [indentLo, indentHi],
    optimizeVolume: true,
    volumeRange: [volumeLo, volumeHi],
  };
};

/** Собрать полный набор конфигов, дополняя отсутствующие дефолтами */
const mergeWithDefaults = (
  genome: BotGenome,
  existing: OrderOptimizationConfig[],
): OrderOptimizationConfig[] => {
  const allOrders: GridOrderGene[] = [genome.baseOrder, ...genome.dcaOrders];
  return allOrders.map((order, idx) => {
    const found = existing.find((c) => c.index === idx);
    return found ?? buildDefaultConfig(idx, order);
  });
};

// ═══════════════════════════════════════════════════════════════════
// ROW TYPE
// ═══════════════════════════════════════════════════════════════════

interface OrderRow {
  key: number;
  label: string;
  currentIndent: number;
  currentVolume: number;
  config: OrderOptimizationConfig;
}

// ═══════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════

const OrderSettingsModal: React.FC<OrderSettingsModalProps> = ({
  open,
  onClose,
  genome,
  orderConfigs,
  onSave,
  showTakeProfit,
  takeProfitConfig,
  onSaveTakeProfit,
}) => {
  const [configs, setConfigs] = useState<OrderOptimizationConfig[]>([]);
  const [tpConfig, setTpConfig] = useState<TakeProfitOptimizationConfig>({
    locked: false,
    valueRange: [0.1, 10],
  });

  // При открытии / смене генома — пересобираем конфиги
  useEffect(() => {
    if (!genome) return;
    setConfigs(mergeWithDefaults(genome, orderConfigs));

    // TP defaults
    if (takeProfitConfig) {
      setTpConfig(takeProfitConfig);
    } else {
      const tpVal = genome.takeProfit.value;
      setTpConfig({
        locked: false,
        valueRange: [Math.max(0.1, +(tpVal * 0.5).toFixed(2)), +(tpVal * 2).toFixed(2)],
      });
    }
  }, [genome, orderConfigs, takeProfitConfig, open]);

  // Обновить одну настройку по индексу
  const updateConfig = useCallback(
    (index: number, patch: Partial<OrderOptimizationConfig>) => {
      setConfigs((prev) =>
        prev.map((c) => (c.index === index ? { ...c, ...patch } : c)),
      );
    },
    [],
  );

  const handleSave = useCallback(() => {
    onSave(configs);
    if (showTakeProfit) {
      onSaveTakeProfit(tpConfig);
    }
    onClose();
  }, [configs, tpConfig, onSave, onSaveTakeProfit, showTakeProfit, onClose]);

  // Быстрые действия
  const lockAll = useCallback(() => {
    setConfigs((prev) =>
      prev.map((c) => ({ ...c, locked: true, optimizeIndent: false, optimizeVolume: false })),
    );
  }, []);

  const unlockAll = useCallback(() => {
    setConfigs((prev) =>
      prev.map((c) => ({ ...c, locked: false, optimizeIndent: true, optimizeVolume: true })),
    );
  }, []);

  // Данные для таблицы
  const rows: OrderRow[] = useMemo(() => {
    if (!genome) return [];
    const allOrders: GridOrderGene[] = [genome.baseOrder, ...genome.dcaOrders];
    return allOrders.map((order, idx) => ({
      key: idx,
      label: idx === 0 ? 'Базовый ордер' : `DCA #${idx}`,
      currentIndent: order.indent,
      currentVolume: order.volume,
      config: configs.find((c) => c.index === idx) ?? buildDefaultConfig(idx, order),
    }));
  }, [genome, configs]);

  // ═════════════ COLUMNS ═════════════

  const columns: ColumnsType<OrderRow> = useMemo(
    () => [
      {
        title: 'Ордер',
        dataIndex: 'label',
        width: 120,
        render: (label: string, row: OrderRow) => (
          <Space>
            <Text strong>{label}</Text>
            {row.config.locked && (
              <Tag color="red" style={{ margin: 0 }}>🔒</Tag>
            )}
          </Space>
        ),
      },
      {
        title: 'Текущие',
        children: [
          {
            title: 'Отступ %',
            dataIndex: 'currentIndent',
            width: 80,
            align: 'center' as const,
            render: (v: number) => <Text type="secondary">{v.toFixed(2)}</Text>,
          },
          {
            title: 'Объём %',
            dataIndex: 'currentVolume',
            width: 80,
            align: 'center' as const,
            render: (v: number) => <Text type="secondary">{v.toFixed(2)}</Text>,
          },
        ],
      },
      {
        title: 'Фикс.',
        width: 60,
        align: 'center' as const,
        render: (_: unknown, row: OrderRow) => (
          <Tooltip title={row.config.locked ? 'Разблокировать' : 'Зафиксировать (не мутировать)'}>
            <Button
              type={row.config.locked ? 'primary' : 'default'}
              danger={row.config.locked}
              size="small"
              icon={row.config.locked ? <LockOutlined /> : <UnlockOutlined />}
              onClick={() =>
                updateConfig(row.key, {
                  locked: !row.config.locked,
                  optimizeIndent: row.config.locked ? true : false,
                  optimizeVolume: row.config.locked ? true : false,
                })
              }
            />
          </Tooltip>
        ),
      },
      {
        title: 'Отступ',
        children: [
          {
            title: 'Опт.',
            width: 50,
            align: 'center' as const,
            render: (_: unknown, row: OrderRow) => (
              <Checkbox
                checked={row.config.optimizeIndent}
                disabled={row.config.locked}
                onChange={(e) => updateConfig(row.key, { optimizeIndent: e.target.checked })}
              />
            ),
          },
          {
            title: 'Мин %',
            width: 80,
            align: 'center' as const,
            render: (_: unknown, row: OrderRow) => (
              <InputNumber
                size="small"
                min={0}
                max={row.config.indentRange[1]}
                step={0.1}
                value={row.config.indentRange[0]}
                disabled={row.config.locked || !row.config.optimizeIndent}
                onChange={(v) =>
                  updateConfig(row.key, {
                    indentRange: [v ?? 0, row.config.indentRange[1]],
                  })
                }
                style={{ width: 70 }}
              />
            ),
          },
          {
            title: 'Макс %',
            width: 80,
            align: 'center' as const,
            render: (_: unknown, row: OrderRow) => (
              <InputNumber
                size="small"
                min={row.config.indentRange[0]}
                max={50}
                step={0.1}
                value={row.config.indentRange[1]}
                disabled={row.config.locked || !row.config.optimizeIndent}
                onChange={(v) =>
                  updateConfig(row.key, {
                    indentRange: [row.config.indentRange[0], v ?? 50],
                  })
                }
                style={{ width: 70 }}
              />
            ),
          },
        ],
      },
      {
        title: 'Объём',
        children: [
          {
            title: 'Опт.',
            width: 50,
            align: 'center' as const,
            render: (_: unknown, row: OrderRow) => (
              <Checkbox
                checked={row.config.optimizeVolume}
                disabled={row.config.locked}
                onChange={(e) => updateConfig(row.key, { optimizeVolume: e.target.checked })}
              />
            ),
          },
          {
            title: 'Мин %',
            width: 80,
            align: 'center' as const,
            render: (_: unknown, row: OrderRow) => (
              <InputNumber
                size="small"
                min={1}
                max={row.config.volumeRange[1]}
                step={0.5}
                value={row.config.volumeRange[0]}
                disabled={row.config.locked || !row.config.optimizeVolume}
                onChange={(v) =>
                  updateConfig(row.key, {
                    volumeRange: [v ?? 1, row.config.volumeRange[1]],
                  })
                }
                style={{ width: 70 }}
              />
            ),
          },
          {
            title: 'Макс %',
            width: 80,
            align: 'center' as const,
            render: (_: unknown, row: OrderRow) => (
              <InputNumber
                size="small"
                min={row.config.volumeRange[0]}
                max={100}
                step={0.5}
                value={row.config.volumeRange[1]}
                disabled={row.config.locked || !row.config.optimizeVolume}
                onChange={(v) =>
                  updateConfig(row.key, {
                    volumeRange: [row.config.volumeRange[0], v ?? 100],
                  })
                }
                style={{ width: 70 }}
              />
            ),
          },
        ],
      },
    ],
    [updateConfig],
  );

  return (
    <Modal
      title="🔧 Настройки оптимизации ордеров"
      open={open}
      onCancel={onClose}
      width={900}
      footer={[
        <Button key="lockAll" onClick={lockAll} icon={<LockOutlined />}>
          Зафиксировать все
        </Button>,
        <Button key="unlockAll" onClick={unlockAll} icon={<UnlockOutlined />}>
          Разблокировать все
        </Button>,
        <Button key="cancel" onClick={onClose}>
          Отмена
        </Button>,
        <Button key="save" type="primary" onClick={handleSave}>
          Сохранить
        </Button>,
      ]}
    >
      <Text type="secondary" style={{ display: 'block', marginBottom: 12 }}>
        Для каждого ордера можно зафиксировать текущие значения (🔒) или задать
        диапазон, в котором оптимизатор будет искать лучший вариант.
      </Text>

      <Table<OrderRow>
        columns={columns}
        dataSource={rows}
        pagination={false}
        size="small"
        bordered
        scroll={{ x: 800 }}
      />

      {/* Секция тейк-профита */}
      {showTakeProfit && genome && (
        <div style={{ marginTop: 16, padding: '12px 16px', border: '1px solid #303030', borderRadius: 8 }}>
          <Space align="center" style={{ marginBottom: 8 }}>
            <Text strong>Тейк-профит</Text>
            <Tag color={genome.takeProfit.type === 'PNL' ? 'blue' : 'green'}>
              {genome.takeProfit.type}
            </Tag>
            <Text type="secondary">текущее: {genome.takeProfit.value.toFixed(2)}</Text>
          </Space>
          <Space size="middle">
            <Tooltip title={tpConfig.locked ? 'Разблокировать' : 'Зафиксировать (не мутировать)'}>
              <Button
                type={tpConfig.locked ? 'primary' : 'default'}
                danger={tpConfig.locked}
                size="small"
                icon={tpConfig.locked ? <LockOutlined /> : <UnlockOutlined />}
                onClick={() => setTpConfig((prev) => ({ ...prev, locked: !prev.locked }))}
              />
            </Tooltip>
            <Space size={4}>
              <Text type="secondary">Мин:</Text>
              <InputNumber
                size="small"
                min={0.01}
                max={tpConfig.valueRange[1]}
                step={0.1}
                value={tpConfig.valueRange[0]}
                disabled={tpConfig.locked}
                onChange={(v) =>
                  setTpConfig((prev) => ({ ...prev, valueRange: [v ?? 0.1, prev.valueRange[1]] }))
                }
                style={{ width: 80 }}
              />
            </Space>
            <Space size={4}>
              <Text type="secondary">Макс:</Text>
              <InputNumber
                size="small"
                min={tpConfig.valueRange[0]}
                max={100}
                step={0.1}
                value={tpConfig.valueRange[1]}
                disabled={tpConfig.locked}
                onChange={(v) =>
                  setTpConfig((prev) => ({ ...prev, valueRange: [prev.valueRange[0], v ?? 10] }))
                }
                style={{ width: 80 }}
              />
            </Space>
          </Space>
        </div>
      )}
    </Modal>
  );
};

export default OrderSettingsModal;
