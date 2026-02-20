/**
 * Walk-forward разбиение периода на train/test окна
 *
 * Поддерживает два режима:
 * 1. sliding — скользящее окно фиксированного размера
 * 2. expanding — растущий train с фиксированным test
 */

import type { WalkForwardConfig, WalkForwardWindow } from '../types/autoOptimizer';

const MS_IN_DAY = 86_400_000;

/**
 * Парсинг ISO даты в timestamp (ms)
 */
const toMs = (iso: string): number => new Date(iso).getTime();

/**
 * Timestamp → Full ISO date string (e.g. 2023-02-16T00:00:00.000Z)
 */
const toISO = (ms: number): string => new Date(ms).toISOString();

/**
 * Разбить период на walk-forward окна.
 *
 * При sliding=true: разбивает на `windowCount` равных блоков, в каждом
 * train занимает `trainRatio`, test — остаток.
 *
 * При sliding=false (anchored): train всегда начинается с начала периода
 * и растёт с каждым окном; test — следующий фиксированный блок.
 */
export const buildWalkForwardWindows = (config: WalkForwardConfig): WalkForwardWindow[] => {
  const startMs = toMs(config.periodFrom);
  const endMs = toMs(config.periodTo);
  const totalMs = endMs - startMs;

  if (totalMs <= 0) {
    throw new Error('Конец периода должен быть позже начала.');
  }

  if (config.windowCount < 1 || config.windowCount > 20) {
    throw new Error('Кол-во окон должно быть от 1 до 20.');
  }

  if (config.trainRatio < 0.5 || config.trainRatio > 0.95) {
    throw new Error('Доля train должна быть от 0.5 до 0.95.');
  }

  const windows: WalkForwardWindow[] = [];

  if (config.sliding) {
    // Скользящее: разбиваем период на overlapping сегменты
    // Шаг сдвига = (total - windowSize) / (windowCount - 1) , если windows > 1
    const windowSize = totalMs / config.windowCount;
    const step = config.windowCount > 1
      ? (totalMs - windowSize) / (config.windowCount - 1)
      : 0;

    for (let i = 0; i < config.windowCount; i++) {
      const winStart = startMs + i * step;
      const winEnd = winStart + windowSize;
      const trainEnd = winStart + windowSize * config.trainRatio;

      windows.push({
        index: i,
        trainFrom: toISO(winStart),
        trainTo: toISO(trainEnd),
        testFrom: toISO(trainEnd),
        testTo: toISO(Math.min(winEnd, endMs)),
        trainDays: Math.round((trainEnd - winStart) / MS_IN_DAY),
        testDays: Math.round((Math.min(winEnd, endMs) - trainEnd) / MS_IN_DAY),
      });
    }
  } else {
    // Anchored (expanding train): train всегда с начала, test — скользит
    const testBlockSize = totalMs * (1 - config.trainRatio) / config.windowCount;

    for (let i = 0; i < config.windowCount; i++) {
      const testStart = startMs + totalMs * config.trainRatio +
        i * (totalMs * (1 - config.trainRatio)) / config.windowCount;
      const testEnd = testStart + testBlockSize;

      // Train: от начала до начала test
      const trainStart = startMs;
      const trainEnd = testStart;

      windows.push({
        index: i,
        trainFrom: toISO(trainStart),
        trainTo: toISO(trainEnd),
        testFrom: toISO(testStart),
        testTo: toISO(Math.min(testEnd, endMs)),
        trainDays: Math.round((trainEnd - trainStart) / MS_IN_DAY),
        testDays: Math.round((Math.min(testEnd, endMs) - testStart) / MS_IN_DAY),
      });
    }
  }

  return windows;
};

/**
 * Итого дней в периоде
 */
export const periodTotalDays = (from: string, to: string): number => {
  return Math.round((toMs(to) - toMs(from)) / MS_IN_DAY);
};

/**
 * Красивое представление окна для логов
 */
export const formatWindow = (w: WalkForwardWindow): string => {
  return `WF#${w.index + 1}: train ${w.trainFrom}→${w.trainTo} (${w.trainDays}д), test ${w.testFrom}→${w.testTo} (${w.testDays}д)`;
};
