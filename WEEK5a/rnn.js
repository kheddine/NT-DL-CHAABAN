// rnn.js
// Defines and trains a multi-output RNN for binary up/down classification (10 stocks × 3 days).

export class RNNModel {
  constructor(inputShape, outputSize) {
    this.model = null;
    this.inputShape = inputShape;
    this.outputSize = outputSize;
  }

  buildModel() {
    this.model = tf.sequential({
      layers: [
        tf.layers.simpleRNN({ units: 64, returnSequences: true, inputShape: this.inputShape }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.simpleRNN({ units: 32 }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' })
      ]
    });

    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return this.model;
  }

  async train(X_train, y_train, X_test, y_test, epochs = 30, batchSize = 32) {
    if (!this.model) this.buildModel();
    await tf.setBackend('webgl');
    await tf.ready();

    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: [X_test, y_test],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress = document.getElementById("trainingProgress");
          const status = document.getElementById("status");
          if (progress) progress.value = ((epoch + 1) / epochs) * 100;
          if (status)
            status.textContent =
              `Epoch ${epoch + 1}/${epochs} | loss: ${logs.loss.toFixed(4)} | acc: ${(logs.binaryAccuracy * 100).toFixed(2)}%`;
          await tf.nextFrame();
        }
      }
    });
  }

  predict(X) {
    return this.model.predict(X);
  }

  evaluatePerStock(yTrue, yPred, symbols, horizon = 3) {
    const trueArr = yTrue.arraySync();
    const predArr = yPred.arraySync();
    const results = {}, details = {};
    symbols.forEach((s, si) => {
      let correct = 0, total = 0;
      const preds = [];
      for (let i = 0; i < trueArr.length; i++) {
        for (let h = 0; h < horizon; h++) {
          const idx = si * horizon + h;
          const t = trueArr[i][idx];
          const p = predArr[i][idx] > 0.5 ? 1 : 0;
          preds.push({ true: t, pred: p, correct: t === p });
          if (t === p) correct++;
          total++;
        }
      }
      results[s] = correct / total;
      details[s] = preds;
    });
    return { stockAccuracies: results, stockPredictions: details };
  }

  dispose() {
    if (this.model) this.model.dispose();
  }
}
