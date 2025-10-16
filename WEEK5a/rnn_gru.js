/**
 * rnn_gru.js
 * Defines the hybrid SimpleRNN + GRU model along with helper methods for
 * backend initialisation, training, and evaluation analytics.
 */

import { DataConstants } from './data-loader.js';

const DEFAULT_EPOCHS = 50;
const DEFAULT_BATCH_SIZE = 64;

/**
 * Attempt to initialise the WebGL backend, falling back to CPU when required.
 * @param {(message: string) => void} statusCb
 */
export async function ensureBackend(statusCb) {
  const desiredBackends = ['webgl', 'cpu'];
  for (const backend of desiredBackends) {
    try {
      await tf.setBackend(backend);
      await tf.ready();
      statusCb?.(`Using TensorFlow.js backend: ${backend}`);
      return backend;
    } catch (err) {
      console.warn(`Failed to initialise backend ${backend}`, err);
    }
  }
  throw new Error('Unable to initialise any TensorFlow.js backend.');
}

/**
 * Builds the hybrid SimpleRNN + GRU model for a given number of stocks.
 * @param {number} numStocks
 */
export function createRNNGRUModel(numStocks) {
  const featureSize = numStocks * DataConstants.FEATURES_PER_STOCK;
  const outputSize = numStocks * DataConstants.FORECAST_HORIZON;

  const model = tf.sequential({ layers: [] });
  model.add(
    tf.layers.simpleRNN({
      units: 64,
      returnSequences: true,
      inputShape: [DataConstants.INPUT_WINDOW, featureSize],
      kernelInitializer: 'glorotUniform',
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.gru({ units: 64, returnSequences: false }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputSize, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['binaryAccuracy'],
  });

  return model;
}

/**
 * Train the provided model.
 * @param {tf.Sequential} model
 * @param {{
 *  X_train: tf.Tensor,
 *  y_train: tf.Tensor,
 *  X_val?: tf.Tensor|null,
 *  y_val?: tf.Tensor|null
 * }} dataset
 * @param {{
 *  epochs?: number,
 *  batchSize?: number,
 *  onEpochBegin?: (epoch: number, logs: tf.Logs) => void,
 *  onEpochEnd?: (epoch: number, logs: tf.Logs) => void,
 * }} options
 */
function createEarlyStoppingWithRestore(model, options) {
  const { patience = 6, monitor = 'loss', mode = 'min' } = options ?? {};
  const compare =
    mode === 'max'
      ? (a, b) => a > b
      : (a, b) => a < b;

  let bestMetricValue = mode === 'max' ? -Infinity : Infinity;
  let bestWeights = null;
  let bestEpoch = -1;
  let wait = 0;
  let warnedMissingMetric = false;

  const cloneWeights = (weights) => weights.map((tensor) => tensor.clone());
  const disposeWeights = (weights) => {
    if (!weights) {
      return;
    }
    for (const tensor of weights) {
      tensor.dispose();
    }
  };

  return {
    onTrainBegin() {
      bestMetricValue = mode === 'max' ? -Infinity : Infinity;
      wait = 0;
      warnedMissingMetric = false;
      bestEpoch = -1;
      disposeWeights(bestWeights);
      bestWeights = null;
    },
    onEpochEnd(epoch, logs) {
      const metricValue = logs?.[monitor];

      if (metricValue == null) {
        if (!warnedMissingMetric) {
          console.warn(
            `Early stopping metric "${monitor}" was not found in training logs.`
          );
          warnedMissingMetric = true;
        }
        return;
      }

      if (compare(metricValue, bestMetricValue)) {
        disposeWeights(bestWeights);
        bestWeights = cloneWeights(model.getWeights());
        bestMetricValue = metricValue;
        bestEpoch = epoch;
        wait = 0;
        return;
      }

      wait += 1;
      if (wait >= patience) {
        console.info(
          `Stopping early at epoch ${epoch + 1}. Best metric (${monitor}) was ${bestMetricValue} at epoch ${bestEpoch + 1}.`
        );
        model.stopTraining = true;
      }
    },
    onTrainEnd() {
      if (!bestWeights) {
        return;
      }

      const weightCopies = cloneWeights(bestWeights);
      model.setWeights(weightCopies);
      disposeWeights(weightCopies);
      disposeWeights(bestWeights);
      bestWeights = null;
    },
  };
}

export async function trainModel(model, dataset, options = {}) {
  const { X_train, y_train, X_val, y_val } = dataset;
  if (!X_train || !y_train) {
    throw new Error('Training tensors are missing.');
  }

  const inputShape = X_train.shape;
  if (inputShape.length !== 3 || inputShape[1] !== DataConstants.INPUT_WINDOW) {
    throw new Error('Training data does not have the expected 3D shape.');
  }

  const epochs = options.epochs ?? DEFAULT_EPOCHS;
  const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;

  const callbacks = [];
  callbacks.push(
    createEarlyStoppingWithRestore(model, {
      patience: 6,
      monitor: X_val && y_val ? 'val_binaryAccuracy' : 'binaryAccuracy',
      mode: 'max',
    })
  );

  if (options.onEpochBegin || options.onEpochEnd) {
    callbacks.push({
      onEpochBegin: options.onEpochBegin,
      onEpochEnd: options.onEpochEnd,
    });
  }

  const history = await model.fit(X_train, y_train, {
    epochs,
    batchSize,
    shuffle: true,
    validationData: X_val && y_val ? [X_val, y_val] : undefined,
    callbacks,
  });

  return history;
}

/**
 * Evaluate the model on the test set and compute per-stock accuracies.
 * @param {tf.Sequential} model
 * @param {{ X_test: tf.Tensor, y_test: tf.Tensor, symbols: string[] }} dataset
 */
export async function evaluateModel(model, dataset) {
  const { X_test, y_test, symbols } = dataset;
  if (!X_test || !y_test) {
    throw new Error('Testing tensors are missing.');
  }

  const numStocks = symbols.length;
  const horizon = DataConstants.FORECAST_HORIZON;

  const predsTensor = model.predict(X_test, { batchSize: DEFAULT_BATCH_SIZE });
  const [preds, truths] = await Promise.all([predsTensor.array(), y_test.array()]);
  predsTensor.dispose();

  const stockAccuracies = {};
  const stockTimeline = {};
  const correctCounts = new Array(numStocks).fill(0);
  const totalCounts = new Array(numStocks).fill(0);

  for (let stockIdx = 0; stockIdx < numStocks; stockIdx++) {
    const symbol = symbols[stockIdx];
    stockTimeline[symbol] = [];
  }

  for (let sampleIdx = 0; sampleIdx < preds.length; sampleIdx++) {
    for (let stockIdx = 0; stockIdx < numStocks; stockIdx++) {
      for (let step = 0; step < horizon; step++) {
        const idx = stockIdx * horizon + step;
        const predProb = preds[sampleIdx][idx];
        const truth = truths[sampleIdx][idx];
        const pred = predProb >= 0.5 ? 1 : 0;
        const correct = pred === truth ? 1 : 0;
        correctCounts[stockIdx] += correct;
        totalCounts[stockIdx] += 1;
        stockTimeline[symbols[stockIdx]].push({ truth, pred, correct });
      }
    }
  }

  for (let stockIdx = 0; stockIdx < numStocks; stockIdx++) {
    stockAccuracies[symbols[stockIdx]] =
      totalCounts[stockIdx] === 0 ? 0 : correctCounts[stockIdx] / totalCounts[stockIdx];
  }

  return {
    stockAccuracies,
    stockPredictions: stockTimeline,
  };
}
