import { Typography, theme } from 'antd';
import { useEffect, useRef } from 'react';
import type { PortfolioLogEntry } from '../../types/portfolioOptimizer';
import type { RunLogEntry } from './runSession';

const { Text } = Typography;

/** Насколько близко к низу (px) нужно быть, чтобы журнал прокручивался за новыми записями. */
const AUTOSCROLL_THRESHOLD_PX = 48;

interface PortfolioLogViewerProps {
  logs: RunLogEntry[];
}

/** Журнал запуска (последние 300 записей) с автопрокруткой, пока пользователь внизу списка. */
const PortfolioLogViewer = ({ logs }: PortfolioLogViewerProps) => {
  const { token } = theme.useToken();
  const containerRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  const levelColors: Record<PortfolioLogEntry['level'], string> = {
    info: token.colorText,
    success: token.colorSuccess,
    warning: token.colorWarning,
    error: token.colorError,
  };

  // biome-ignore lint/correctness/useExhaustiveDependencies: прокрутка должна срабатывать на каждую новую запись журнала
  useEffect(() => {
    const container = containerRef.current;
    if (container && stickToBottomRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [logs]);

  return (
    <div
      ref={containerRef}
      role="log"
      aria-live="polite"
      onScroll={(event) => {
        const target = event.currentTarget;
        stickToBottomRef.current =
          target.scrollHeight - target.scrollTop - target.clientHeight < AUTOSCROLL_THRESHOLD_PX;
      }}
      style={{
        maxHeight: 320,
        overflowY: 'auto',
        background: token.colorFillQuaternary,
        border: `1px solid ${token.colorBorderSecondary}`,
        padding: 12,
        borderRadius: token.borderRadius,
        fontFamily: token.fontFamilyCode,
        fontSize: 12,
      }}
    >
      {logs.length === 0 ? <Text type="secondary">Журнал появится после запуска…</Text> : null}
      {logs.map((entry) => (
        <div key={entry.id} style={{ marginBottom: 4 }}>
          <Text type="secondary">{new Date(entry.timestamp).toLocaleTimeString('ru-RU')}</Text>{' '}
          <Text style={{ color: levelColors[entry.level] }}>{entry.message}</Text>
        </div>
      ))}
    </div>
  );
};

export default PortfolioLogViewer;
