import { Col, Flex, Progress, Row, Statistic, Tag, Typography } from 'antd';
import type { PortfolioOptimizerProgress } from '../../types/portfolioOptimizer';
import { formatDateTime, formatDuration, formatEta, RUN_STATUS_LABELS, runProgressPercent } from './runView';

const { Text } = Typography;

interface RunProgressPanelProps {
  progress: PortfolioOptimizerProgress;
  /** Оценка общего числа бэктестов запуска (максимум) — для полосы прогресса. */
  expectedBacktests: number;
  now: number;
}

const progressStatus = (status: PortfolioOptimizerProgress['status']) => {
  if (status === 'running') {
    return 'active' as const;
  }
  if (status === 'error') {
    return 'exception' as const;
  }
  if (status === 'completed') {
    return 'success' as const;
  }
  return 'normal' as const;
};

/** Прогресс запуска: статус, поколение, бэктесты, очередь планировщика и оценка окончания. */
const RunProgressPanel = ({ progress, expectedBacktests, now }: RunProgressPanelProps) => {
  const status = RUN_STATUS_LABELS[progress.status];
  const eta = formatEta(progress.estimatedEndAt, now);
  const elapsed = progress.startedAt !== null ? formatDuration(now - progress.startedAt) : null;

  return (
    <Flex vertical gap={12}>
      <Flex gap={8} align="center" wrap>
        <Tag color={status.tone}>{status.text}</Tag>
        {progress.startedAt !== null ? (
          <Text type="secondary">
            Старт: {formatDateTime(progress.startedAt)}
            {progress.status === 'running' || progress.status === 'paused' ? ` · идёт ${elapsed}` : ''}
          </Text>
        ) : null}
      </Flex>
      <Progress
        percent={runProgressPercent(progress, expectedBacktests)}
        status={progressStatus(progress.status)}
        aria-label="Прогресс запуска"
      />
      <Row gutter={[16, 12]}>
        <Col xs={12} md={8} xl={4}>
          <Statistic title="Поколение" value={progress.generation} suffix={`/ ${progress.totalGenerations}`} />
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Statistic
            title="Бэктестов создано"
            value={progress.backtestsSubmitted}
            suffix={`/ ${progress.backtestsLimit}`}
          />
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Statistic title="В работе" value={progress.inFlight} />
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Statistic title="В очереди" value={progress.queued} />
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Statistic title="Оценено кандидатов" value={progress.candidatesEvaluated} />
        </Col>
        <Col xs={24} md={16} xl={4}>
          <Statistic title="Окончание (оценка)" value={eta ?? '—'} valueStyle={{ fontSize: 14 }} />
        </Col>
      </Row>
    </Flex>
  );
};

export default RunProgressPanel;
