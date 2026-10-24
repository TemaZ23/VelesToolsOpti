import { AppstoreAddOutlined, CopyOutlined, RobotOutlined, TrophyOutlined } from '@ant-design/icons';
import { Button, Empty, Popconfirm, Space, Table, Tag, Tooltip, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useMemo } from 'react';
import {
  buildOptimizedBotName,
  CANDIDATE_PHASE_LABELS,
  type CandidateActionTarget,
  type CandidateRow,
  compareNullable,
  compareRowMetric,
  formatNumber,
} from './runView';

const { Text } = Typography;

const PAGE_SIZE = 20;

interface CandidatesTableProps {
  rows: CandidateRow[];
  botName: string;
  quoteCurrency: string;
  /** Создание бота доступно (расширение подключено). */
  canCreateBot: boolean;
  /** Создаётся ли сейчас бот для кандидата (одновременно может создаваться несколько). */
  isCreatingBot: (candidateId: string) => boolean;
  onCreateGroup: (target: CandidateActionTarget) => void;
  onCopyJson: (target: CandidateActionTarget) => void;
  onCreateBot: (target: CandidateActionTarget) => void;
}

const ScreenMark = ({ row }: { row: CandidateRow }) =>
  row.metricsStage === 'screen' ? (
    <Tooltip title="Пока посчитана только отборочная корзина (меньше монет — не сравнивайте с полной)">
      <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>
        отб.
      </Text>
    </Tooltip>
  ) : null;

/** Значение метрики: по отборочной корзине — приглушённо и с пометкой «отб.». */
const MetricCell = ({
  row,
  value,
  digits = 2,
}: {
  row: CandidateRow;
  value: number | null | undefined;
  digits?: number;
}) => (
  <span>
    {row.metricsStage === 'screen' ? (
      <Text type="secondary">{formatNumber(value, digits)}</Text>
    ) : (
      formatNumber(value, digits)
    )}
    <ScreenMark row={row} />
  </span>
);

const CandidatesTable = ({
  rows,
  botName,
  quoteCurrency,
  canCreateBot,
  isCreatingBot,
  onCreateGroup,
  onCopyJson,
  onCreateBot,
}: CandidatesTableProps) => {
  const columns = useMemo<ColumnsType<CandidateRow>>(
    () => [
      {
        title: 'Кандидат',
        key: 'label',
        width: 130,
        fixed: 'left',
        render: (_value, row) => (
          <Space size={4} wrap>
            <Text strong={row.isBest}>{row.label}</Text>
            {row.isBest ? (
              <Tag color="gold" icon={<TrophyOutlined />}>
                лучший
              </Tag>
            ) : null}
          </Space>
        ),
      },
      {
        title: 'Пок.',
        dataIndex: 'generation',
        key: 'generation',
        width: 64,
        align: 'right',
        sorter: (left, right) => left.generation - right.generation,
      },
      {
        title: 'Изменено',
        key: 'changes',
        width: 96,
        align: 'right',
        sorter: (left, right) => left.changes.length - right.changes.length,
        render: (_value, row) =>
          row.changes.length === 0 ? (
            <Text type="secondary">—</Text>
          ) : (
            <Tooltip
              title={
                <div>
                  {row.changes.map((change) => (
                    <div key={change}>{change}</div>
                  ))}
                </div>
              }
            >
              <Tag style={{ cursor: 'help', marginInlineEnd: 0 }}>{row.changes.length}</Tag>
            </Tooltip>
          ),
      },
      {
        title: 'Отбор',
        key: 'screenScore',
        width: 84,
        align: 'right',
        sorter: (left, right) => compareNullable(left.screenScore, right.screenScore),
        render: (_value, row) => formatNumber(row.screenScore, 3),
      },
      {
        title: 'Полная',
        key: 'fullScore',
        width: 84,
        align: 'right',
        sorter: (left, right) => compareNullable(left.fullScore, right.fullScore),
        render: (_value, row) => <Text strong={row.fullScore !== null}>{formatNumber(row.fullScore, 3)}</Text>,
      },
      {
        title: `P&L, ${quoteCurrency}`,
        key: 'profit',
        width: 120,
        align: 'right',
        sorter: (left, right) => compareRowMetric(left, right, (metrics) => metrics.totalProfit),
        render: (_value, row) => {
          const profit = row.metrics?.totalProfit ?? null;
          return (
            <span>
              <Text type={profit === null ? 'secondary' : profit >= 0 ? 'success' : 'danger'}>
                {formatNumber(profit)}
              </Text>
              <ScreenMark row={row} />
            </span>
          );
        },
      },
      {
        title: 'МПУ',
        key: 'mae',
        width: 100,
        align: 'right',
        sorter: (left, right) => compareRowMetric(left, right, (metrics) => metrics.maxConcurrentMae),
        render: (_value, row) => <MetricCell row={row} value={row.metrics?.maxConcurrentMae} />,
      },
      {
        title: 'P&L/риск',
        key: 'pnlToRisk',
        width: 96,
        align: 'right',
        sorter: (left, right) => compareRowMetric(left, right, (metrics) => metrics.pnlToRisk),
        render: (_value, row) => <MetricCell row={row} value={row.metrics?.pnlToRisk} />,
      },
      {
        title: 'Сделок',
        key: 'deals',
        width: 84,
        align: 'right',
        sorter: (left, right) => compareRowMetric(left, right, (metrics) => metrics.totalDeals),
        render: (_value, row) => <MetricCell row={row} value={row.metrics?.totalDeals} digits={0} />,
      },
      {
        title: 'Статус',
        key: 'phase',
        width: 170,
        render: (_value, row) => {
          const label = CANDIDATE_PHASE_LABELS[row.phase];
          const tag = <Tag color={label.tone}>{label.text}</Tag>;
          return row.error ? <Tooltip title={row.error}>{tag}</Tooltip> : tag;
        },
      },
      {
        title: 'Действия',
        key: 'actions',
        width: 128,
        fixed: 'right',
        render: (_value, row) => {
          const backtestsCount = row.backtests?.ids.length ?? 0;
          const botNameForRow = buildOptimizedBotName(botName, row.label);
          return (
            <Space size={4}>
              <Tooltip
                title={
                  row.backtests
                    ? `Создать группу бэктестов (${row.backtests.stage === 'full' ? 'полная корзина' : 'отбор'}: ${backtestsCount})`
                    : 'Бэктестов кандидата пока нет'
                }
              >
                <Button
                  size="small"
                  icon={<AppstoreAddOutlined />}
                  aria-label={`Группа бэктестов ${row.label}`}
                  disabled={!row.backtests}
                  onClick={() => onCreateGroup(row)}
                />
              </Tooltip>
              <Tooltip title="Копировать JSON стратегии">
                <Button
                  size="small"
                  icon={<CopyOutlined />}
                  aria-label={`JSON стратегии ${row.label}`}
                  onClick={() => onCopyJson(row)}
                />
              </Tooltip>
              <Popconfirm
                title="Создать бота в Veles?"
                description={`Будет создан бот «${botNameForRow}» с параметрами кандидата ${row.label}.`}
                okText="Создать"
                cancelText="Отмена"
                disabled={row.isBase || !canCreateBot}
                onConfirm={() => onCreateBot(row)}
              >
                <Tooltip
                  title={
                    row.isBase ? 'Это исходный бот' : canCreateBot ? 'Создать бота' : 'Расширение не подключено к Veles'
                  }
                >
                  <Button
                    size="small"
                    icon={<RobotOutlined />}
                    aria-label={`Создать бота ${row.label}`}
                    disabled={row.isBase || !canCreateBot}
                    loading={isCreatingBot(row.candidate.id)}
                  />
                </Tooltip>
              </Popconfirm>
            </Space>
          );
        },
      },
    ],
    [botName, canCreateBot, isCreatingBot, onCopyJson, onCreateBot, onCreateGroup, quoteCurrency],
  );

  if (rows.length === 0) {
    return <Empty description="Кандидаты появятся после старта оптимизации." />;
  }

  return (
    <Table<CandidateRow>
      rowKey="key"
      size="small"
      columns={columns}
      dataSource={rows}
      pagination={rows.length > PAGE_SIZE ? { pageSize: PAGE_SIZE, showSizeChanger: false } : false}
      scroll={{ x: 1180 }}
    />
  );
};

export default CandidatesTable;
