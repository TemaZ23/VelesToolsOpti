import { CheckSquareOutlined, StopOutlined, UndoOutlined } from '@ant-design/icons';
import { Button, Checkbox, Empty, Flex, InputNumber, Table, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useMemo } from 'react';
import { formatParamValue } from '../../lib/botParams';
import type { ParamSpec } from '../../types/portfolioOptimizer';
import {
  countParamValues,
  disableGrid,
  enableAllThresholds,
  minParamStep,
  type ParamBoundField,
  setParamEnabled,
  setParamsEnabledWhere,
  updateParamBound,
} from './formState';

const { Text } = Typography;

interface ParamsTableProps {
  params: ParamSpec[];
  disabled: boolean;
  onChange: (params: ParamSpec[]) => void;
  onReset: () => void;
}

interface GroupInfo {
  /** Сколько строк объединяет ячейка группы (0 — строка внутри группы). */
  rowSpan: number;
  enabled: number;
  total: number;
}

/** Группы идут подряд (порядок extractParams): ячейка группы объединяет строки одной группы. */
const buildGroupInfo = (params: ParamSpec[]): Map<string, GroupInfo> => {
  const result = new Map<string, GroupInfo>();
  let start = 0;
  for (let index = 1; index <= params.length; index += 1) {
    if (index < params.length && params[index].group === params[start].group) {
      continue;
    }
    const members = params.slice(start, index);
    const enabled = members.filter((param) => param.enabled).length;
    members.forEach((param, offset) => {
      result.set(param.id, { rowSpan: offset === 0 ? members.length : 0, enabled, total: members.length });
    });
    start = index;
  }
  return result;
};

const ParamsTable = ({ params, disabled, onChange, onReset }: ParamsTableProps) => {
  const groupInfo = useMemo(() => buildGroupInfo(params), [params]);
  const enabledCount = params.filter((param) => param.enabled).length;

  const columns = useMemo<ColumnsType<ParamSpec>>(() => {
    const boundInput = (record: ParamSpec, field: ParamBoundField) => {
      const isStep = field === 'step';
      return (
        <InputNumber
          size="small"
          style={{ width: '100%' }}
          value={record[field]}
          min={field === 'max' ? record.base : isStep ? minParamStep(record.decimals) : undefined}
          max={field === 'min' ? record.base : undefined}
          step={isStep ? minParamStep(record.decimals) : record.step}
          disabled={disabled}
          aria-label={`${record.label}: ${field === 'min' ? 'минимум' : field === 'max' ? 'максимум' : 'шаг'}`}
          onChange={(value) => onChange(updateParamBound(params, record.id, field, value))}
        />
      );
    };

    return [
      {
        title: 'Группа',
        key: 'group',
        width: 130,
        onCell: (record) => ({ rowSpan: groupInfo.get(record.id)?.rowSpan ?? 1 }),
        render: (_value, record) => {
          const info = groupInfo.get(record.id);
          const allEnabled = info !== undefined && info.enabled === info.total;
          return (
            <Checkbox
              checked={allEnabled}
              indeterminate={info !== undefined && info.enabled > 0 && !allEnabled}
              disabled={disabled}
              onChange={(event) =>
                onChange(setParamsEnabledWhere(params, (param) => param.group === record.group, event.target.checked))
              }
            >
              {record.group}
            </Checkbox>
          );
        },
      },
      {
        title: 'Вкл',
        key: 'enabled',
        width: 56,
        align: 'center',
        render: (_value, record) => (
          <Checkbox
            checked={record.enabled}
            disabled={disabled}
            aria-label={`Оптимизировать «${record.label}»`}
            onChange={(event) => onChange(setParamEnabled(params, record.id, event.target.checked))}
          />
        ),
      },
      {
        title: 'Параметр',
        dataIndex: 'label',
        key: 'label',
        render: (label: string, record) => <Text type={record.enabled ? undefined : 'secondary'}>{label}</Text>,
      },
      {
        title: 'База',
        key: 'base',
        width: 90,
        align: 'right',
        render: (_value, record) => <Text strong>{formatParamValue(record.base, record.decimals)}</Text>,
      },
      { title: 'Мин', key: 'min', width: 110, render: (_value, record) => boundInput(record, 'min') },
      { title: 'Макс', key: 'max', width: 110, render: (_value, record) => boundInput(record, 'max') },
      { title: 'Шаг', key: 'step', width: 100, render: (_value, record) => boundInput(record, 'step') },
      {
        title: 'Значений',
        key: 'values',
        width: 90,
        align: 'right',
        render: (_value, record) => <Text type="secondary">{countParamValues(record)}</Text>,
      },
    ];
  }, [disabled, groupInfo, onChange, params]);

  if (params.length === 0) {
    return <Empty description="Загрузите базового бота, чтобы увидеть его параметры." />;
  }

  return (
    <Flex vertical gap={12}>
      <Flex gap={8} wrap align="center" justify="space-between">
        <Flex gap={8} wrap>
          <Button
            size="small"
            icon={<CheckSquareOutlined />}
            disabled={disabled}
            onClick={() => onChange(enableAllThresholds(params))}
          >
            Включить все пороги
          </Button>
          <Button
            size="small"
            icon={<StopOutlined />}
            disabled={disabled}
            onClick={() => onChange(disableGrid(params))}
          >
            Выключить сетку
          </Button>
          <Button size="small" icon={<UndoOutlined />} disabled={disabled} onClick={onReset}>
            Сбросить
          </Button>
        </Flex>
        <Text type="secondary">
          В поиске {enabledCount} из {params.length} параметров
        </Text>
      </Flex>
      <Table<ParamSpec>
        rowKey="id"
        size="small"
        bordered
        columns={columns}
        dataSource={params}
        pagination={false}
        scroll={{ x: 760, y: 520 }}
      />
      <Text type="secondary" style={{ fontSize: 12 }}>
        Мутация сдвигает параметр на 1–2 шага в пределах [мин, макс]. Отступы ордеров остаются возрастающими, сумма
        объёмов — как у базового бота. Остальные поля бота не меняются.
      </Text>
    </Flex>
  );
};

export default ParamsTable;
