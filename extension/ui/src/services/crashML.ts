/**
 * Browser-based ML для анализа crash'ей
 * 
 * Реализация ML алгоритмов на чистом TypeScript без внешних зависимостей.
 * 
 * Алгоритмы:
 * - Logistic Regression (с L2 регуляризацией)
 * - Decision Tree (CART)
 * - Random Forest (ensemble of trees)
 * 
 * Особенности:
 * - Walk-forward validation для временных рядов
 * - Работа с дисбалансом классов (class weights)
 * - Feature importance
 * - Rule extraction
 */

// ═══════════════════════════════════════════════════════════════════════════════
// ТИПЫ
// ═══════════════════════════════════════════════════════════════════════════════

export interface MLDataset {
  X: number[][];           // Features matrix [n_samples, n_features]
  y: number[];             // Target vector [n_samples]
  featureNames: string[];  // Feature names
  timestamps: number[];    // Timestamps for time-based splitting
}

export interface TrainTestSplit {
  X_train: number[][];
  y_train: number[];
  X_test: number[][];
  y_test: number[];
  trainPeriod: { from: string; to: string };
  testPeriod: { from: string; to: string };
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  confusionMatrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative' | 'mixed';
}

export interface ExtractedRule {
  conditions: Array<{
    feature: string;
    operator: '>' | '<' | '>=' | '<=';
    threshold: number;
  }>;
  probability: number;
  support: number;
  lift: number;
  confidence: number;
}

export interface MLModelResult {
  name: string;
  metrics: ModelMetrics;
  featureImportance: FeatureImportanceItem[];
  predictions: number[];
  probabilities: number[];
}

export interface MLAnalysisResult {
  models: MLModelResult[];
  bestModel: MLModelResult;
  walkForwardResults: Array<{
    fold: number;
    trainPeriod: { from: string; to: string };
    testPeriod: { from: string; to: string };
    metrics: Record<string, ModelMetrics>;
  }>;
  extractedRules: ExtractedRule[];
  featureImportance: FeatureImportanceItem[];
  validationInfo: {
    trainSize: number;
    testSize: number;
    nSplits: number;
    stepSize: number;
  };
  datasetInfo: {
    totalSamples: number;
    crashSamples: number;
    nonCrashSamples: number;
    crashRate: number;
  };
  aggregatedMetrics: Record<string, {
    meanAuc: number;
    stdAuc: number;
    meanPrecision: number;
    meanRecall: number;
  }>;
}

export interface MLConfig {
  crashThresholdPct: number;
  crashWindowBars: number;
  trainMonths: number;
  testMonths: number;
  minSamplesLeaf: number;
  maxTreeDepth: number;
  nTrees: number;
  learningRate: number;
  regularization: number;
}

export const DEFAULT_ML_CONFIG: MLConfig = {
  crashThresholdPct: 5,
  crashWindowBars: 48,
  trainMonths: 24,
  testMonths: 6,
  minSamplesLeaf: 20,
  maxTreeDepth: 6,
  nTrees: 50,
  learningRate: 0.1,
  regularization: 0.01,
};

// ═══════════════════════════════════════════════════════════════════════════════
// УТИЛИТЫ
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Среднее значение
 */
const mean = (arr: number[]): number => {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
};

/**
 * Стандартное отклонение
 */
const std = (arr: number[]): number => {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
};

/**
 * Сигмоида
 */
const sigmoid = (x: number): number => {
  if (x > 500) return 1;
  if (x < -500) return 0;
  return 1 / (1 + Math.exp(-x));
};

/**
 * Dot product
 */
const dot = (a: number[], b: number[]): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
};

/**
 * Transpose matrix
 */
const transpose = (matrix: number[][]): number[][] => {
  if (matrix.length === 0) return [];
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result: number[][] = Array(cols).fill(null).map(() => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
};

/**
 * Нормализация данных (StandardScaler)
 */
const standardize = (X: number[][]): { X_scaled: number[][]; means: number[]; stds: number[] } => {
  if (X.length === 0) return { X_scaled: [], means: [], stds: [] };
  
  const nFeatures = X[0].length;
  const means: number[] = [];
  const stds: number[] = [];
  
  // Рассчитываем mean и std для каждого признака
  const transposed = transpose(X);
  for (let j = 0; j < nFeatures; j++) {
    means.push(mean(transposed[j]));
    stds.push(std(transposed[j]) || 1); // Avoid division by zero
  }
  
  // Нормализуем
  const X_scaled = X.map(row => 
    row.map((val, j) => (val - means[j]) / stds[j])
  );
  
  return { X_scaled, means, stds };
};

/**
 * Применить нормализацию к новым данным
 */
const applyStandardization = (X: number[][], means: number[], stds: number[]): number[][] => {
  return X.map(row => 
    row.map((val, j) => (val - means[j]) / stds[j])
  );
};

/**
 * Форматирование timestamp
 */
const formatDate = (ts: number): string => {
  return new Date(ts).toISOString().split('T')[0];
};

// ═══════════════════════════════════════════════════════════════════════════════
// LOGISTIC REGRESSION
// ═══════════════════════════════════════════════════════════════════════════════

interface LogisticRegressionModel {
  weights: number[];
  bias: number;
  means: number[];
  stds: number[];
}

/**
 * Обучение Logistic Regression методом градиентного спуска
 */
const trainLogisticRegression = (
  X: number[][],
  y: number[],
  config: MLConfig,
  maxIterations: number = 1000
): LogisticRegressionModel => {
  const { X_scaled, means, stds } = standardize(X);
  const nSamples = X_scaled.length;
  const nFeatures = X_scaled[0]?.length || 0;
  
  // Инициализация весов
  let weights = new Array(nFeatures).fill(0);
  let bias = 0;
  
  // Class weights для дисбаланса
  const nPositive = y.filter(v => v === 1).length;
  const nNegative = nSamples - nPositive;
  const weightPositive = nSamples / (2 * Math.max(nPositive, 1));
  const weightNegative = nSamples / (2 * Math.max(nNegative, 1));
  
  const lr = config.learningRate;
  const lambda = config.regularization;
  
  for (let iter = 0; iter < maxIterations; iter++) {
    const gradW = new Array(nFeatures).fill(0);
    let gradB = 0;
    
    for (let i = 0; i < nSamples; i++) {
      const z = dot(X_scaled[i], weights) + bias;
      const pred = sigmoid(z);
      const classWeight = y[i] === 1 ? weightPositive : weightNegative;
      const error = (pred - y[i]) * classWeight;
      
      for (let j = 0; j < nFeatures; j++) {
        gradW[j] += error * X_scaled[i][j];
      }
      gradB += error;
    }
    
    // Update with regularization
    for (let j = 0; j < nFeatures; j++) {
      weights[j] -= lr * (gradW[j] / nSamples + lambda * weights[j]);
    }
    bias -= lr * (gradB / nSamples);
  }
  
  return { weights, bias, means, stds };
};

/**
 * Предсказание Logistic Regression
 */
const predictLogisticRegression = (
  model: LogisticRegressionModel,
  X: number[][]
): { predictions: number[]; probabilities: number[] } => {
  const X_scaled = applyStandardization(X, model.means, model.stds);
  
  const probabilities = X_scaled.map(row => {
    const z = dot(row, model.weights) + model.bias;
    return sigmoid(z);
  });
  
  const predictions = probabilities.map(p => p >= 0.5 ? 1 : 0);
  
  return { predictions, probabilities };
};

/**
 * Feature importance из весов Logistic Regression
 */
const getLogisticRegressionImportance = (
  model: LogisticRegressionModel,
  featureNames: string[]
): FeatureImportanceItem[] => {
  return model.weights
    .map((w, i) => ({
      feature: featureNames[i],
      importance: Math.abs(w),
      direction: (w > 0 ? 'positive' : 'negative') as 'positive' | 'negative',
    }))
    .sort((a, b) => b.importance - a.importance);
};

// ═══════════════════════════════════════════════════════════════════════════════
// DECISION TREE (CART)
// ═══════════════════════════════════════════════════════════════════════════════

interface TreeNode {
  isLeaf: boolean;
  prediction?: number;
  probability?: number;
  samples?: number;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

interface DecisionTreeModel {
  root: TreeNode;
  featureImportances: number[];
  means: number[];
  stds: number[];
}

/**
 * Расчёт Gini impurity
 */
const giniImpurity = (y: number[]): number => {
  if (y.length === 0) return 0;
  const p = y.filter(v => v === 1).length / y.length;
  return 2 * p * (1 - p);
};

/**
 * Расчёт weighted Gini для split
 */
const weightedGini = (yLeft: number[], yRight: number[]): number => {
  const nTotal = yLeft.length + yRight.length;
  if (nTotal === 0) return 0;
  return (
    (yLeft.length / nTotal) * giniImpurity(yLeft) +
    (yRight.length / nTotal) * giniImpurity(yRight)
  );
};

/**
 * Найти лучший split для узла
 */
const findBestSplit = (
  X: number[][],
  y: number[],
  featureIndices: number[],
  minSamplesLeaf: number
): { featureIndex: number; threshold: number; gain: number } | null => {
  const nSamples = X.length;
  if (nSamples < 2 * minSamplesLeaf) return null;
  
  const currentGini = giniImpurity(y);
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  
  for (const featureIndex of featureIndices) {
    // Получаем уникальные значения признака
    const values = X.map(row => row[featureIndex]).sort((a, b) => a - b);
    const uniqueValues = [...new Set(values)];
    
    // Пробуем разные пороги
    for (let i = 0; i < uniqueValues.length - 1; i++) {
      const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
      
      const leftIndices: number[] = [];
      const rightIndices: number[] = [];
      
      for (let j = 0; j < nSamples; j++) {
        if (X[j][featureIndex] <= threshold) {
          leftIndices.push(j);
        } else {
          rightIndices.push(j);
        }
      }
      
      if (leftIndices.length < minSamplesLeaf || rightIndices.length < minSamplesLeaf) {
        continue;
      }
      
      const yLeft = leftIndices.map(idx => y[idx]);
      const yRight = rightIndices.map(idx => y[idx]);
      
      const splitGini = weightedGini(yLeft, yRight);
      const gain = currentGini - splitGini;
      
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = featureIndex;
        bestThreshold = threshold;
      }
    }
  }
  
  if (bestFeature === -1) return null;
  
  return { featureIndex: bestFeature, threshold: bestThreshold, gain: bestGain };
};

/**
 * Построение дерева рекурсивно
 */
const buildTree = (
  X: number[][],
  y: number[],
  depth: number,
  config: MLConfig,
  featureImportances: number[]
): TreeNode => {
  const nSamples = y.length;
  const nPositive = y.filter(v => v === 1).length;
  const probability = nPositive / nSamples;
  
  // Leaf conditions
  if (
    depth >= config.maxTreeDepth ||
    nSamples < 2 * config.minSamplesLeaf ||
    nPositive === 0 ||
    nPositive === nSamples
  ) {
    return {
      isLeaf: true,
      prediction: probability >= 0.5 ? 1 : 0,
      probability,
      samples: nSamples,
    };
  }
  
  // Find best split
  const nFeatures = X[0].length;
  const featureIndices = Array.from({ length: nFeatures }, (_, i) => i);
  const split = findBestSplit(X, y, featureIndices, config.minSamplesLeaf);
  
  if (!split || split.gain < 0.001) {
    return {
      isLeaf: true,
      prediction: probability >= 0.5 ? 1 : 0,
      probability,
      samples: nSamples,
    };
  }
  
  // Update feature importance
  featureImportances[split.featureIndex] += split.gain * nSamples;
  
  // Split data
  const leftIndices: number[] = [];
  const rightIndices: number[] = [];
  
  for (let i = 0; i < nSamples; i++) {
    if (X[i][split.featureIndex] <= split.threshold) {
      leftIndices.push(i);
    } else {
      rightIndices.push(i);
    }
  }
  
  const X_left = leftIndices.map(i => X[i]);
  const y_left = leftIndices.map(i => y[i]);
  const X_right = rightIndices.map(i => X[i]);
  const y_right = rightIndices.map(i => y[i]);
  
  return {
    isLeaf: false,
    featureIndex: split.featureIndex,
    threshold: split.threshold,
    samples: nSamples,
    left: buildTree(X_left, y_left, depth + 1, config, featureImportances),
    right: buildTree(X_right, y_right, depth + 1, config, featureImportances),
  };
};

/**
 * Обучение Decision Tree
 */
const trainDecisionTree = (
  X: number[][],
  y: number[],
  config: MLConfig
): DecisionTreeModel => {
  const { X_scaled, means, stds } = standardize(X);
  const nFeatures = X_scaled[0]?.length || 0;
  const featureImportances = new Array(nFeatures).fill(0);
  
  const root = buildTree(X_scaled, y, 0, config, featureImportances);
  
  // Normalize importances
  const totalImportance = featureImportances.reduce((a, b) => a + b, 0) || 1;
  const normalizedImportances = featureImportances.map(v => v / totalImportance);
  
  return { root, featureImportances: normalizedImportances, means, stds };
};

/**
 * Предсказание одного примера
 */
const predictSingleTree = (node: TreeNode, x: number[]): { prediction: number; probability: number } => {
  if (node.isLeaf) {
    return { prediction: node.prediction!, probability: node.probability! };
  }
  
  if (x[node.featureIndex!] <= node.threshold!) {
    return predictSingleTree(node.left!, x);
  } else {
    return predictSingleTree(node.right!, x);
  }
};

/**
 * Предсказание Decision Tree
 */
const predictDecisionTree = (
  model: DecisionTreeModel,
  X: number[][]
): { predictions: number[]; probabilities: number[] } => {
  const X_scaled = applyStandardization(X, model.means, model.stds);
  
  const results = X_scaled.map(x => predictSingleTree(model.root, x));
  
  return {
    predictions: results.map(r => r.prediction),
    probabilities: results.map(r => r.probability),
  };
};

// ═══════════════════════════════════════════════════════════════════════════════
// RANDOM FOREST
// ═══════════════════════════════════════════════════════════════════════════════

interface RandomForestModel {
  trees: DecisionTreeModel[];
  means: number[];
  stds: number[];
  featureImportances: number[];
}

/**
 * Bootstrap sampling
 */
const bootstrapSample = (
  X: number[][],
  y: number[],
  sampleSize: number
): { X_sample: number[][]; y_sample: number[] } => {
  const X_sample: number[][] = [];
  const y_sample: number[] = [];
  
  for (let i = 0; i < sampleSize; i++) {
    const idx = Math.floor(Math.random() * X.length);
    X_sample.push([...X[idx]]);
    y_sample.push(y[idx]);
  }
  
  return { X_sample, y_sample };
};

/**
 * Обучение Random Forest
 */
const trainRandomForest = (
  X: number[][],
  y: number[],
  config: MLConfig
): RandomForestModel => {
  const { X_scaled, means, stds } = standardize(X);
  const nFeatures = X_scaled[0]?.length || 0;
  const trees: DecisionTreeModel[] = [];
  const aggregatedImportances = new Array(nFeatures).fill(0);
  
  for (let t = 0; t < config.nTrees; t++) {
    // Bootstrap sample
    const { X_sample, y_sample } = bootstrapSample(X_scaled, y, X_scaled.length);
    
    // Train tree (with already scaled data)
    const treeConfig = { ...config, maxTreeDepth: Math.min(config.maxTreeDepth, 5) };
    const featureImportances = new Array(nFeatures).fill(0);
    const root = buildTree(X_sample, y_sample, 0, treeConfig, featureImportances);
    
    const totalImportance = featureImportances.reduce((a, b) => a + b, 0) || 1;
    const normalizedImportances = featureImportances.map(v => v / totalImportance);
    
    trees.push({
      root,
      featureImportances: normalizedImportances,
      means: [], // Already scaled
      stds: [],
    });
    
    // Aggregate importances
    for (let j = 0; j < nFeatures; j++) {
      aggregatedImportances[j] += normalizedImportances[j];
    }
  }
  
  // Normalize aggregated importances
  const totalAgg = aggregatedImportances.reduce((a, b) => a + b, 0) || 1;
  const finalImportances = aggregatedImportances.map(v => v / totalAgg);
  
  return { trees, means, stds, featureImportances: finalImportances };
};

/**
 * Предсказание Random Forest
 */
const predictRandomForest = (
  model: RandomForestModel,
  X: number[][]
): { predictions: number[]; probabilities: number[] } => {
  const X_scaled = applyStandardization(X, model.means, model.stds);
  
  const probabilities = X_scaled.map(x => {
    let sumProb = 0;
    for (const tree of model.trees) {
      const { probability } = predictSingleTree(tree.root, x);
      sumProb += probability;
    }
    return sumProb / model.trees.length;
  });
  
  const predictions = probabilities.map(p => p >= 0.5 ? 1 : 0);
  
  return { predictions, probabilities };
};

// ═══════════════════════════════════════════════════════════════════════════════
// МЕТРИКИ
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Расчёт метрик классификации
 */
const calculateMetrics = (
  y_true: number[],
  y_pred: number[],
  y_proba: number[]
): ModelMetrics => {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  
  for (let i = 0; i < y_true.length; i++) {
    if (y_pred[i] === 1 && y_true[i] === 1) tp++;
    else if (y_pred[i] === 1 && y_true[i] === 0) fp++;
    else if (y_pred[i] === 0 && y_true[i] === 0) tn++;
    else fn++;
  }
  
  const accuracy = (tp + tn) / (tp + fp + tn + fn) || 0;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * precision * recall / (precision + recall) || 0;
  
  // ROC AUC (trapezoid rule)
  const rocAuc = calculateRocAuc(y_true, y_proba);
  
  return {
    accuracy,
    precision,
    recall,
    f1Score,
    rocAuc,
    confusionMatrix: { tp, fp, tn, fn },
  };
};

/**
 * Расчёт ROC AUC
 */
const calculateRocAuc = (y_true: number[], y_proba: number[]): number => {
  // Sort by probability descending
  const indices = y_proba
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .map(item => item.i);
  
  const sorted_y = indices.map(i => y_true[i]);
  
  const nPositive = y_true.filter(v => v === 1).length;
  const nNegative = y_true.length - nPositive;
  
  if (nPositive === 0 || nNegative === 0) return 0.5;
  
  let tpr = 0;
  let fpr = 0;
  let prevTpr = 0;
  let prevFpr = 0;
  let auc = 0;
  
  for (let i = 0; i < sorted_y.length; i++) {
    if (sorted_y[i] === 1) {
      tpr += 1 / nPositive;
    } else {
      fpr += 1 / nNegative;
      // Trapezoid area
      auc += (tpr + prevTpr) * (fpr - prevFpr) / 2;
      prevTpr = tpr;
      prevFpr = fpr;
    }
  }
  
  return auc;
};

// ═══════════════════════════════════════════════════════════════════════════════
// WALK-FORWARD VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Создание walk-forward splits
 */
const createWalkForwardSplits = (
  dataset: MLDataset,
  config: MLConfig
): TrainTestSplit[] => {
  const splits: TrainTestSplit[] = [];
  
  // Sort by timestamp
  const indices = dataset.timestamps
    .map((ts, i) => ({ ts, i }))
    .sort((a, b) => a.ts - b.ts)
    .map(item => item.i);
  
  const sortedTimestamps = indices.map(i => dataset.timestamps[i]);
  const minTs = sortedTimestamps[0];
  const maxTs = sortedTimestamps[sortedTimestamps.length - 1];
  
  const msPerMonth = 30 * 24 * 60 * 60 * 1000;
  let trainEndTs = minTs + config.trainMonths * msPerMonth;
  
  while (trainEndTs + config.testMonths * msPerMonth <= maxTs) {
    const testEndTs = trainEndTs + config.testMonths * msPerMonth;
    
    const trainIndices = indices.filter((_, i) => sortedTimestamps[i] < trainEndTs);
    const testIndices = indices.filter((_, i) => 
      sortedTimestamps[i] >= trainEndTs && sortedTimestamps[i] < testEndTs
    );
    
    if (trainIndices.length >= 1000 && testIndices.length >= 100) {
      splits.push({
        X_train: trainIndices.map(i => dataset.X[i]),
        y_train: trainIndices.map(i => dataset.y[i]),
        X_test: testIndices.map(i => dataset.X[i]),
        y_test: testIndices.map(i => dataset.y[i]),
        trainPeriod: {
          from: formatDate(minTs),
          to: formatDate(trainEndTs),
        },
        testPeriod: {
          from: formatDate(trainEndTs),
          to: formatDate(testEndTs),
        },
      });
    }
    
    trainEndTs += config.testMonths * msPerMonth;
  }
  
  return splits;
};

// ═══════════════════════════════════════════════════════════════════════════════
// RULE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Извлечение правил из данных
 */
const extractRules = (
  dataset: MLDataset,
  config: MLConfig,
  minSupport: number = 30,
  minLift: number = 1.5
): ExtractedRule[] => {
  const { X, y, featureNames } = dataset;
  const baseCrashRate = mean(y);
  const rules: ExtractedRule[] = [];
  
  // Single-feature rules
  for (let j = 0; j < featureNames.length; j++) {
    const values = X.map(row => row[j]).sort((a, b) => a - b);
    const quantiles = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    
    for (const q of quantiles) {
      const threshold = values[Math.floor(values.length * q)];
      
      // Rule: feature > threshold
      const aboveIndices = X.map((row, i) => row[j] > threshold ? i : -1).filter(i => i >= 0);
      if (aboveIndices.length >= minSupport) {
        const crashes = aboveIndices.filter(i => y[i] === 1).length;
        const prob = crashes / aboveIndices.length;
        const lift = prob / baseCrashRate;
        
        if (lift >= minLift) {
          rules.push({
            conditions: [{ feature: featureNames[j], operator: '>', threshold }],
            probability: prob,
            support: aboveIndices.length,
            lift,
            confidence: prob,
          });
        }
      }
      
      // Rule: feature < threshold
      const belowIndices = X.map((row, i) => row[j] < threshold ? i : -1).filter(i => i >= 0);
      if (belowIndices.length >= minSupport) {
        const crashes = belowIndices.filter(i => y[i] === 1).length;
        const prob = crashes / belowIndices.length;
        const lift = prob / baseCrashRate;
        
        if (lift >= minLift) {
          rules.push({
            conditions: [{ feature: featureNames[j], operator: '<', threshold }],
            probability: prob,
            support: belowIndices.length,
            lift,
            confidence: prob,
          });
        }
      }
    }
  }
  
  // Sort by lift and take top rules
  rules.sort((a, b) => b.lift - a.lift);
  const topRules = rules.slice(0, 15);
  
  // Try combining top rules
  const combinedRules: ExtractedRule[] = [];
  
  for (let i = 0; i < Math.min(topRules.length, 10); i++) {
    for (let j = i + 1; j < Math.min(topRules.length, 10); j++) {
      const rule1 = topRules[i];
      const rule2 = topRules[j];
      
      if (rule1.conditions[0].feature === rule2.conditions[0].feature) continue;
      
      // Find samples matching both conditions
      const matchingIndices = X.map((row, idx) => {
        const cond1 = rule1.conditions[0];
        const cond2 = rule2.conditions[0];
        const feat1Idx = featureNames.indexOf(cond1.feature);
        const feat2Idx = featureNames.indexOf(cond2.feature);
        
        const match1 = cond1.operator === '>' 
          ? row[feat1Idx] > cond1.threshold 
          : row[feat1Idx] < cond1.threshold;
        const match2 = cond2.operator === '>' 
          ? row[feat2Idx] > cond2.threshold 
          : row[feat2Idx] < cond2.threshold;
        
        return match1 && match2 ? idx : -1;
      }).filter(i => i >= 0);
      
      if (matchingIndices.length >= minSupport) {
        const crashes = matchingIndices.filter(i => y[i] === 1).length;
        const prob = crashes / matchingIndices.length;
        const lift = prob / baseCrashRate;
        
        if (lift >= minLift * 1.3) { // Higher threshold for combined rules
          combinedRules.push({
            conditions: [...rule1.conditions, ...rule2.conditions],
            probability: prob,
            support: matchingIndices.length,
            lift,
            confidence: prob,
          });
        }
      }
    }
  }
  
  // Merge and sort
  const allRules = [...topRules, ...combinedRules];
  allRules.sort((a, b) => b.lift - a.lift);
  
  return allRules.slice(0, 20);
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN ANALYSIS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

export interface MLProgressCallback {
  (stage: string, progress: number, message: string): void;
}

/**
 * Запуск полного ML анализа
 */
export const runMLAnalysis = async (
  dataset: MLDataset,
  config: MLConfig,
  onProgress?: MLProgressCallback
): Promise<MLAnalysisResult> => {
  const progress = onProgress || (() => {});
  
  progress('preparing', 0, 'Подготовка данных...');
  
  // Create walk-forward splits
  const splits = createWalkForwardSplits(dataset, config);
  
  if (splits.length === 0) {
    throw new Error('Недостаточно данных для walk-forward validation');
  }
  
  progress('training', 10, `Создано ${splits.length} fold(s) для валидации`);
  
  const modelNames = ['Logistic Regression', 'Decision Tree', 'Random Forest'];
  const allFoldResults: MLAnalysisResult['walkForwardResults'] = [];
  const aggregatedByModel: Record<string, { aucs: number[]; precisions: number[]; recalls: number[] }> = {};
  
  for (const name of modelNames) {
    aggregatedByModel[name] = { aucs: [], precisions: [], recalls: [] };
  }
  
  let lastResults: MLModelResult[] = [];
  
  // Train on each fold
  for (let foldIdx = 0; foldIdx < splits.length; foldIdx++) {
    const split = splits[foldIdx];
    const foldProgress = 10 + (foldIdx / splits.length) * 70;
    
    progress('training', foldProgress, `Fold ${foldIdx + 1}/${splits.length}: Обучение моделей...`);
    
    const foldMetrics: Record<string, ModelMetrics> = {};
    const foldResults: MLModelResult[] = [];
    
    // Logistic Regression
    const lrModel = trainLogisticRegression(split.X_train, split.y_train, config);
    const lrPred = predictLogisticRegression(lrModel, split.X_test);
    const lrMetrics = calculateMetrics(split.y_test, lrPred.predictions, lrPred.probabilities);
    foldMetrics['Logistic Regression'] = lrMetrics;
    aggregatedByModel['Logistic Regression'].aucs.push(lrMetrics.rocAuc);
    aggregatedByModel['Logistic Regression'].precisions.push(lrMetrics.precision);
    aggregatedByModel['Logistic Regression'].recalls.push(lrMetrics.recall);
    
    foldResults.push({
      name: 'Logistic Regression',
      metrics: lrMetrics,
      featureImportance: getLogisticRegressionImportance(lrModel, dataset.featureNames),
      predictions: lrPred.predictions,
      probabilities: lrPred.probabilities,
    });
    
    // Decision Tree
    const dtModel = trainDecisionTree(split.X_train, split.y_train, config);
    const dtPred = predictDecisionTree(dtModel, split.X_test);
    const dtMetrics = calculateMetrics(split.y_test, dtPred.predictions, dtPred.probabilities);
    foldMetrics['Decision Tree'] = dtMetrics;
    aggregatedByModel['Decision Tree'].aucs.push(dtMetrics.rocAuc);
    aggregatedByModel['Decision Tree'].precisions.push(dtMetrics.precision);
    aggregatedByModel['Decision Tree'].recalls.push(dtMetrics.recall);
    
    foldResults.push({
      name: 'Decision Tree',
      metrics: dtMetrics,
      featureImportance: dtModel.featureImportances.map((imp, i) => ({
        feature: dataset.featureNames[i],
        importance: imp,
        direction: 'mixed' as const,
      })).sort((a, b) => b.importance - a.importance),
      predictions: dtPred.predictions,
      probabilities: dtPred.probabilities,
    });
    
    // Random Forest
    const rfModel = trainRandomForest(split.X_train, split.y_train, config);
    const rfPred = predictRandomForest(rfModel, split.X_test);
    const rfMetrics = calculateMetrics(split.y_test, rfPred.predictions, rfPred.probabilities);
    foldMetrics['Random Forest'] = rfMetrics;
    aggregatedByModel['Random Forest'].aucs.push(rfMetrics.rocAuc);
    aggregatedByModel['Random Forest'].precisions.push(rfMetrics.precision);
    aggregatedByModel['Random Forest'].recalls.push(rfMetrics.recall);
    
    foldResults.push({
      name: 'Random Forest',
      metrics: rfMetrics,
      featureImportance: rfModel.featureImportances.map((imp, i) => ({
        feature: dataset.featureNames[i],
        importance: imp,
        direction: 'mixed' as const,
      })).sort((a, b) => b.importance - a.importance),
      predictions: rfPred.predictions,
      probabilities: rfPred.probabilities,
    });
    
    allFoldResults.push({
      fold: foldIdx + 1,
      trainPeriod: split.trainPeriod,
      testPeriod: split.testPeriod,
      metrics: foldMetrics,
    });
    
    lastResults = foldResults;
    
    // Allow UI to update
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  
  progress('rules', 85, 'Извлечение правил...');
  
  // Extract rules
  const extractedRules = extractRules(dataset, config);
  
  progress('aggregating', 95, 'Агрегация результатов...');
  
  // Aggregate metrics
  const aggregatedMetrics: Record<string, { meanAuc: number; stdAuc: number; meanPrecision: number; meanRecall: number }> = {};
  
  for (const [name, data] of Object.entries(aggregatedByModel)) {
    aggregatedMetrics[name] = {
      meanAuc: mean(data.aucs),
      stdAuc: std(data.aucs),
      meanPrecision: mean(data.precisions),
      meanRecall: mean(data.recalls),
    };
  }
  
  // Find best model
  const bestModelName = Object.entries(aggregatedMetrics)
    .sort((a, b) => b[1].meanAuc - a[1].meanAuc)[0][0];
  
  const bestModelResult = lastResults.find(m => m.name === bestModelName) || lastResults[0];
  
  // Aggregate feature importance across models
  const featureImportanceMap = new Map<string, number>();
  for (const model of lastResults) {
    for (const fi of model.featureImportance) {
      const current = featureImportanceMap.get(fi.feature) || 0;
      featureImportanceMap.set(fi.feature, current + fi.importance);
    }
  }
  // Normalize
  const totalImportance = Array.from(featureImportanceMap.values()).reduce((a, b) => a + b, 0) || 1;
  const featureImportance: FeatureImportanceItem[] = Array.from(featureImportanceMap.entries())
    .map(([feature, importance]) => ({
      feature,
      importance: importance / totalImportance,
      direction: 'mixed' as const,
    }))
    .sort((a, b) => b.importance - a.importance);
  
  // Dataset info
  const crashSamples = dataset.y.filter(v => v === 1).length;
  const nonCrashSamples = dataset.y.filter(v => v === 0).length;
  
  progress('done', 100, 'Анализ завершён!');
  
  return {
    models: lastResults,
    bestModel: bestModelResult,
    walkForwardResults: allFoldResults,
    extractedRules,
    featureImportance,
    validationInfo: {
      trainSize: splits[0]?.X_train.length || 0,
      testSize: splits[0]?.X_test.length || 0,
      nSplits: splits.length,
      stepSize: Math.floor(config.testMonths * 30 * 24 * 4), // примерно
    },
    datasetInfo: {
      totalSamples: dataset.X.length,
      crashSamples,
      nonCrashSamples,
      crashRate: crashSamples / (crashSamples + nonCrashSamples),
    },
    aggregatedMetrics,
  };
};

/**
 * Подготовка датасета из FeatureBar[]
 */
export const prepareMLDataset = (
  features: Array<Record<string, number | null>>,
  featureColumns: string[],
  targetColumn: string
): MLDataset => {
  // Filter rows with valid target
  const validRows = features.filter(row => row[targetColumn] !== null && row[targetColumn] !== undefined);
  
  const X: number[][] = [];
  const y: number[] = [];
  const timestamps: number[] = [];
  
  for (const row of validRows) {
    const featureVector: number[] = [];
    let hasNull = false;
    
    for (const col of featureColumns) {
      const val = row[col];
      if (val === null || val === undefined || Number.isNaN(val)) {
        hasNull = true;
        break;
      }
      featureVector.push(val);
    }
    
    if (!hasNull) {
      X.push(featureVector);
      y.push(row[targetColumn] as number);
      timestamps.push(row.timestamp as number);
    }
  }
  
  return {
    X,
    y,
    featureNames: featureColumns,
    timestamps,
  };
};
