// rnn.js
// Defines and trains the RNN model.

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
        tf.layers.simpleRNN({ units: 32 }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }),
      ],
    });

    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy'],
    });

    return this.model;
  }

  async train(X_train, y_train, X_test, y_test, epochs = 25, batchSize = 32) {
    if (!this.model) this.buildModel();
    await tf.setBackend('webgl');
    await tf.ready();

    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: [X_test, y_test],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress = document.getElementById('trainingProgress');
          const status = document.getElementById('status');
          if (progress) progress.value = ((epoch + 1) / epochs) * 100;
          if (status)
            status.textContent =
              `Epoch ${epoch + 1}/${epochs} | loss: ${logs.loss.toFixed(4)} | acc: ${(logs.binaryAccuracy * 100).toFixed(1)}%`;
          await tf.nextFrame();
        },
      },
    });
  }

  predict(X) {
    return this.model.predict(X);
  }

  evaluate(yTrue, yPred, symbols, horizon = 3) {
    const t = yTrue.arraySync();
    const p = yPred.arraySync();
    const acc = {}, detail = {};

    symbols.forEach((sym, i) => {
      let correct = 0, total = 0;
      const preds = [];
      for (let s = 0; s < t.length; s++) {
        for (let h = 0; h < horizon; h++) {
          const idx = i * horizon + h;
          const truth = t[s][idx];
          const pred = p[s][idx] > 0.5 ? 1 : 0;
          preds.push({ truth, pred, correct: truth === pred });
          if (truth === pred) correct++;
          total++;
        }
      }
      acc[sym] = correct / total;
      detail[sym] = preds;
    });
    return { stockAccuracies: acc, stockPredictions: detail };
  }
}
