import { DeleteOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { Alert, Button, Flex, Popconfirm, Typography } from 'antd';
import type { PortfolioOptimizerState } from '../../types/portfolioOptimizer';
import type { OtherTabRun } from './runLock';
import { canResumeSavedRun, formatDateTime, savedRunStatusLabel } from './runView';

const { Text } = Typography;

interface SavedRunBannerProps {
  state: PortfolioOptimizerState;
  /** Выполняется ли этот запуск в другой вкладке (блокировка или недавняя активность). */
  elsewhere: OtherTabRun;
  /** Продолжение недоступно (например, расширение не подключено). */
  resumeDisabled: boolean;
  onResume: () => void;
  onClear: () => void;
}

/** Сохранённый запуск: краткие сведения, «Продолжить» и «Удалить». */
const SavedRunBanner = ({ state, elsewhere, resumeDisabled, onResume, onClear }: SavedRunBannerProps) => {
  const locked = elsewhere === 'locked';
  const resumable = canResumeSavedRun(state) || locked;
  const { search } = state.config;
  const status = savedRunStatusLabel(state.status, locked);

  return (
    <Alert
      type={elsewhere === 'none' ? 'info' : 'warning'}
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
          {locked ? (
            <Text type="warning">
              Этот запуск выполняется в другой вкладке расширения — управляйте им там. Продолжить или начать новый
              запуск здесь можно после того, как он остановится или та вкладка закроется.
            </Text>
          ) : null}
          {elsewhere === 'probable' ? (
            <Text type="warning">
              Запуск был активен несколько минут назад — возможно, он выполняется в другой вкладке расширения.
              Продолжайте его здесь, только если та вкладка закрыта, иначе оптимизация пойдёт дважды.
            </Text>
          ) : null}
          <Flex gap={8} wrap>
            {resumable ? (
              <Button
                type="primary"
                size="small"
                icon={<PlayCircleOutlined />}
                disabled={resumeDisabled || locked}
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
              disabled={locked}
            >
              <Button size="small" danger icon={<DeleteOutlined />} disabled={locked}>
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
