import { DeleteOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { Alert, Button, Flex, Popconfirm, Typography } from 'antd';
import type { PortfolioOptimizerState } from '../../types/portfolioOptimizer';
import { canResumeSavedRun, formatDateTime, isProbablyRunningElsewhere, savedRunStatusLabel } from './runView';

const { Text } = Typography;

interface SavedRunBannerProps {
  state: PortfolioOptimizerState;
  now: number;
  /** Продолжение недоступно (например, расширение не подключено). */
  resumeDisabled: boolean;
  onResume: () => void;
  onClear: () => void;
}

/** Сохранённый запуск: краткие сведения, «Продолжить» и «Удалить». */
const SavedRunBanner = ({ state, now, resumeDisabled, onResume, onClear }: SavedRunBannerProps) => {
  const resumable = canResumeSavedRun(state);
  const elsewhere = isProbablyRunningElsewhere(state, now);
  const { search } = state.config;
  const status = savedRunStatusLabel(state.status);

  return (
    <Alert
      type={elsewhere ? 'warning' : 'info'}
      showIcon
      message={resumable ? 'Найден незавершённый запуск портфельного оптимизатора' : 'Последний запуск завершён'}
      description={
        <Flex vertical gap={8}>
          <Text>
            Бот «{state.config.botName}» · корзина {state.config.basket.length} монет · поколение {state.generation} из{' '}
            {search.generations} · бэктестов {state.backtestsSubmitted} из {search.maxBacktests} · статус «
            {status.text.toLowerCase()}»
          </Text>
          <Text type="secondary">Сохранено: {formatDateTime(state.updatedAt)}</Text>
          {elsewhere ? (
            <Text type="warning">
              Запуск обновлялся несколько минут назад — возможно, он выполняется в другой вкладке расширения.
              Продолжайте его здесь, только если та вкладка закрыта, иначе оптимизация пойдёт дважды.
            </Text>
          ) : null}
          <Flex gap={8} wrap>
            {resumable ? (
              <Button
                type="primary"
                size="small"
                icon={<PlayCircleOutlined />}
                disabled={resumeDisabled}
                onClick={onResume}
              >
                Продолжить
              </Button>
            ) : null}
            <Popconfirm
              title="Удалить сохранённый запуск?"
              description="Результаты исчезнут со страницы. Бэктесты и боты в Veles останутся."
              okText="Удалить"
              cancelText="Отмена"
              onConfirm={onClear}
            >
              <Button size="small" danger icon={<DeleteOutlined />}>
                Удалить
              </Button>
            </Popconfirm>
          </Flex>
        </Flex>
      }
    />
  );
};

export default SavedRunBanner;
