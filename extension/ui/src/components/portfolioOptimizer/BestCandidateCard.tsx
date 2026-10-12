import { AppstoreAddOutlined, CopyOutlined, RobotOutlined, TrophyOutlined } from '@ant-design/icons';
import { Alert, Button, Card, Col, Empty, Flex, Popconfirm, Row, Statistic, Table, Tag, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useMemo } from 'react';
import type { FoldMetrics } from '../../types/portfolioOptimizer';
import {
  type BestSummary,
  buildOptimizedBotName,
  type CandidateActionTarget,
  formatNumber,
  formatSigned,
  type MetricComparison,
} from './runView';

const { Text } = Typography;

interface BestCandidateCardProps {
  summary: BestSummary | null;
  botName: string;
  quoteCurrency: string;
  canCreateBot: boolean;
  creatingBotFor: string | null;
  onCreateGroup: (target: CandidateActionTarget) => void;
  onCopyJson: (target: CandidateActionTarget) => void;
  onCreateBot: (target: CandidateActionTarget) => void;
}

interface MetricTileProps {
  title: string;
  comparison: MetricComparison;
  digits?: number;
  showDelta: boolean;
}

const MetricTile = ({ title, comparison, digits = 2, showDelta }: MetricTileProps) => {
  const deltaType = comparison.better === null ? 'secondary' : comparison.better ? 'success' : 'danger';
  const percent = comparison.deltaPercent === null ? '' : ` (${formatSigned(comparison.deltaPercent, 1)}%)`;
  return (
    <Flex vertical gap={2}>
      <Statistic title={title} value={formatNumber(comparison.value, digits)} />
      {showDelta ? (
        <Text type={deltaType} style={{ fontSize: 12 }}>
          {comparison.delta === null
            ? 'нет данных базы'
            : `${formatSigned(comparison.delta, digits)}${percent} к базе (${formatNumber(comparison.base, digits)})`}
        </Text>
      ) : null}
    </Flex>
  );
};

interface FoldRow {
  key: number;
  fold: FoldMetrics;
  base: FoldMetrics | null;
}

const formatFoldPeriod = (fold: FoldMetrics): string => {
  const from = new Date(fold.from).toISOString().slice(0, 10);
  const to = new Date(Math.max(fold.from, fold.to - 1)).toISOString().slice(0, 10);
  return `${from} — ${to}`;
};

const FoldsTable = ({
  folds,
  baseFolds,
  showBase,
}: {
  folds: FoldMetrics[];
  baseFolds: FoldMetrics[];
  showBase: boolean;
}) => {
  const rows = useMemo<FoldRow[]>(
    () => folds.map((fold, index) => ({ key: fold.from, fold, base: baseFolds.at(index) ?? null })),
    [baseFolds, folds],
  );
  const columns = useMemo<ColumnsType<FoldRow>>(
    () => [
      { title: 'Под-период', key: 'period', render: (_value, row) => formatFoldPeriod(row.fold) },
      { title: 'P&L', key: 'profit', align: 'right', render: (_value, row) => formatNumber(row.fold.profit) },
      { title: 'Риск', key: 'risk', align: 'right', render: (_value, row) => formatNumber(row.fold.risk) },
      {
        title: 'P&L/риск',
        key: 'pnlToRisk',
        align: 'right',
        render: (_value, row) => <Text strong>{formatNumber(row.fold.pnlToRisk)}</Text>,
      },
      ...(showBase
        ? [
            {
              title: 'P&L/риск базы',
              key: 'basePnlToRisk',
              align: 'right' as const,
              render: (_value: unknown, row: FoldRow) => formatNumber(row.base?.pnlToRisk),
            },
          ]
        : []),
      { title: 'Сделок', key: 'deals', align: 'right', render: (_value, row) => formatNumber(row.fold.deals, 0) },
    ],
    [showBase],
  );
  return <Table<FoldRow> rowKey="key" size="small" columns={columns} dataSource={rows} pagination={false} />;
};

const BestCandidateCard = ({
  summary,
  botName,
  quoteCurrency,
  canCreateBot,
  creatingBotFor,
  onCreateGroup,
  onCopyJson,
  onCreateBot,
}: BestCandidateCardProps) => {
  if (!summary) {
    return (
      <Card title="Лучший результат" size="small">
        <Empty description="Результат появится после оценки базового бота на полной корзине." />
      </Card>
    );
  }

  const target: CandidateActionTarget = {
    candidate: summary.best,
    label: summary.label,
    isBase: summary.isBase,
    backtests: summary.backtests,
  };
  const showDelta = !summary.isBase;
  const newBotName = buildOptimizedBotName(botName, summary.label);

  return (
    <Card
      size="small"
      title={
        <Flex gap={8} align="center" wrap>
          <TrophyOutlined style={{ color: '#faad14' }} />
          <span>Лучший результат — {summary.label}</span>
          {summary.isBase ? <Tag>базовый бот</Tag> : <Tag color="gold">поколение {summary.best.generation}</Tag>}
        </Flex>
      }
      extra={
        <Flex gap={8} wrap>
          <Button
            size="small"
            icon={<AppstoreAddOutlined />}
            disabled={!summary.backtests}
            onClick={() => onCreateGroup(target)}
          >
            Группа бэктестов
          </Button>
          <Button size="small" icon={<CopyOutlined />} onClick={() => onCopyJson(target)}>
            JSON
          </Button>
          <Popconfirm
            title="Создать бота в Veles?"
            description={`Будет создан бот «${newBotName}» с найденными параметрами.`}
            okText="Создать"
            cancelText="Отмена"
            disabled={summary.isBase || !canCreateBot}
            onConfirm={() => onCreateBot(target)}
          >
            <Button
              size="small"
              type="primary"
              icon={<RobotOutlined />}
              disabled={summary.isBase || !canCreateBot}
              loading={creatingBotFor === summary.best.id}
            >
              Создать бота
            </Button>
          </Popconfirm>
        </Flex>
      }
    >
      <Flex vertical gap={16}>
        {summary.isBase ? (
          <Alert
            type="info"
            showIcon
            message="Пока лучший — базовый бот"
            description="Ни один кандидат ещё не превзошёл базу на полной корзине с учётом минимального улучшения."
          />
        ) : null}
        <Row gutter={[16, 16]}>
          <Col xs={12} md={8} xl={4}>
            <MetricTile title="Score" comparison={summary.score} digits={3} showDelta={showDelta} />
          </Col>
          <Col xs={12} md={8} xl={4}>
            <MetricTile title={`P&L, ${quoteCurrency}`} comparison={summary.profit} showDelta={showDelta} />
          </Col>
          <Col xs={12} md={8} xl={4}>
            <MetricTile title="МПУ" comparison={summary.mae} showDelta={showDelta} />
          </Col>
          <Col xs={12} md={8} xl={4}>
            <MetricTile title="Макс. просадка" comparison={summary.drawdown} showDelta={showDelta} />
          </Col>
          <Col xs={12} md={8} xl={4}>
            <MetricTile title="P&L к риску" comparison={summary.pnlToRisk} showDelta={showDelta} />
          </Col>
          <Col xs={12} md={8} xl={4}>
            <MetricTile title="Сделок" comparison={summary.deals} digits={0} showDelta={showDelta} />
          </Col>
        </Row>
        <Flex vertical gap={4}>
          <Text strong>Изменённые параметры</Text>
          {summary.changes.length === 0 ? (
            <Text type="secondary">Без изменений относительно базового бота.</Text>
          ) : (
            <ul style={{ margin: 0, paddingInlineStart: 20 }}>
              {summary.changes.map((change) => (
                <li key={change}>
                  <Text>{change}</Text>
                </li>
              ))}
            </ul>
          )}
        </Flex>
        {summary.folds.length > 0 ? (
          <Flex vertical gap={4}>
            <Text strong>Под-периоды (полная корзина)</Text>
            <FoldsTable folds={summary.folds} baseFolds={summary.baseFolds} showBase={showDelta} />
          </Flex>
        ) : null}
      </Flex>
    </Card>
  );
};

export default BestCandidateCard;
