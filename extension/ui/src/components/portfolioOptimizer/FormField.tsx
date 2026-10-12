import { QuestionCircleOutlined } from '@ant-design/icons';
import { Flex, Tooltip, Typography } from 'antd';
import type { ReactNode } from 'react';

const { Text } = Typography;

interface FormFieldProps {
  label: ReactNode;
  /** Подсказка по значку «?» рядом с подписью. */
  hint?: ReactNode;
  /** Пояснение под полем. */
  extra?: ReactNode;
  children: ReactNode;
}

/** Поле формы: подпись (с подсказкой), элемент ввода и пояснение. */
const FormField = ({ label, hint, extra, children }: FormFieldProps) => (
  <Flex vertical gap={4}>
    <Text type="secondary">
      {label}
      {hint ? (
        <Tooltip title={hint}>
          <QuestionCircleOutlined style={{ marginLeft: 6 }} aria-label="Подсказка" />
        </Tooltip>
      ) : null}
    </Text>
    {children}
    {extra ? (
      <Text type="secondary" style={{ fontSize: 12 }}>
        {extra}
      </Text>
    ) : null}
  </Flex>
);

export default FormField;
