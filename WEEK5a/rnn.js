// rnn.js
// Builds, trains, and evaluates an RNN with early stopping

export class RNNModel {
  constructor(inputShape, outputSize) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.model = null;
  }

  buildModel() {
    this.model = tf.sequential({
      layers: [
        tf.layers.simpleRNN({ units: 64, returnSequences: true, inputShape: this.inputShape }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.simpleRNN({ units: 64 }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' })
      ]
    });

    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
  }

  async train(X_train, y_train, X_val, y_val, epochs = 50, batchSize = 64) {
    if (!this.model) this.buildModel();
    await tf.setBackend('webgl');
    await tf.ready();

    const earlyStop = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: 6,
      restoreBestWeights: true
    });

    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: [X_val, y_val],
      shuffle: true,
      callbacks: [
        earlyStop,
        {
          onEpochEnd: async (epoch, logs) => {
            const progress = document.getElementById('trainingProgress');
            const status = document.getElementById('status');
            if (progress) progress.value = ((epoch + 1) / epochs) * 100;
            if (status) status.textContent =
              `Epoch ${epoch + 1}/${epochs} | loss: ${logs.loss.toFixed(4)} | acc: ${(logs.binaryAccuracy * 100).toFixed(1)}% | val_acc: ${(logs.val_binaryAccuracy * 100).toFixed(1)}%`;
            await tf.nextFrame();
          }
        }
      ]
    });
  }

  predict(X) {
    return this.model.predict(X);
  }

  evaluate(yTrue, yPred, symbols, horizon = 3) {
    const t = yTrue.arraySync();
    const p = yPred.arraySync();
    const acc = {}, detail = {};

    symbols.forEach((sym, si) => {
      let correct = 0, total = 0;
      const preds = [];
      for (let s = 0; s < t.length; s++) {
        for (let h = 0; h < horizon; h++) {
          const idx = si * horizon + h;
          const truth = t[s][idx];
          const pred = p[s][idx] > 0.5 ? 1 : 0;
          preds.push({ truth, pred, correct: truth === pred });
          if (truth === pred) correct++;
          total++;
        }
      }
      acc[sym] = total ? correct / total : 0;
      detail[sym] = preds;
    });

    return { stockAccuracies: acc, stockPredictions: detail };
  }
}
