// cnn_gru.js
// CNN + GRU hybrid: Conv1D over time (padding SAME) -> GRU -> Dense.
// Works with input [batch, 12, 20] and outputs [batch, 30].

export class CNN_GRU_Model {
  constructor(inputShape, outputSize) {
    this.inputShape = inputShape;   // [12, 20]
    this.outputSize = outputSize;   // 30 (10 stocks * 3 days)
    this.model = null;
  }

  async ensureBackend() {
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      console.log("TF backend:", tf.getBackend());
    } catch (e) {
      console.warn("WebGL unavailable; falling back to CPU.", e);
      await tf.setBackend('cpu');
      await tf.ready();
    }
  }

  buildModel() {
    // Conv1D expects [time, channels] in tfjs; here time=12, channels=20
    this.model = tf.sequential({
      layers: [
        tf.layers.conv1d({
          inputShape: this.inputShape,
          filters: 64,
          kernelSize: 3,
          padding: 'same',
          activation: 'relu'
        }),
        tf.layers.dropout({ rate: 0.2 }),
        // Conv1D outputs [batch, time, filters] => perfect for GRU
        tf.layers.gru({ units: 64 }),
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

    return this.model;
  }

  async train(Xtr, ytr, Xv, yv, epochs = 50, batchSize = 64) {
    console.log("Train:", Xtr.shape, ytr.shape, "| Val:", Xv.shape, yv.shape);
    await this.ensureBackend();
    if (!this.model) this.buildModel();

    const hasVal = (Xv?.shape?.[0] || 0) > 0 && (yv?.shape?.[0] || 0) > 0;
    const valData = hasVal ? [Xv, yv] : null;
    if (!hasVal) console.warn("No validation split; training without val.");

    const early = tf.callbacks.earlyStopping({
      monitor: hasVal ? 'val_loss' : 'loss',
      patience: 6,
      restoreBestWeights: true
    });

    const uiCb = {
      onEpochEnd: async (epoch, logs) => {
        const p = document.getElementById('trainingProgress');
        const s = document.getElementById('status');
        if (p) p.value = ((epoch + 1) / epochs) * 100;
        if (s) s.textContent =
          `Epoch ${epoch + 1}/${epochs} | loss ${logs.loss.toFixed(4)} | acc ${(logs.binaryAccuracy * 100).toFixed(1)}%` +
          (hasVal && logs.val_binaryAccuracy != null ? ` | val ${(logs.val_binaryAccuracy * 100).toFixed(1)}%` : '');
        await tf.nextFrame();
      }
    };

    console.log("🚀 Starting model.fit");
    const hist = await this.model.fit(Xtr, ytr, {
      epochs,
      batchSize,
      shuffle: true,
      validationData: valData,
      callbacks: [early, uiCb]
    });
    console.log("✅ Training finished");

    return hist;
  }

  predict(X) {
    const out = this.model.predict(X);
    return Array.isArray(out) ? out[0] : out;
  }

  evaluate(yTrue, yPred, symbols, horizon = 3) {
    const t = yTrue.arraySync();
    const p = yPred.arraySync();
    const stockAccuracies = {};
    const stockPredictions = {};

    symbols.forEach((sym, si) => {
      let correct = 0, total = 0;
      const preds = [];
      for (let i = 0; i < t.length; i++) {
        for (let h = 0; h < horizon; h++) {
          const idx = si * horizon + h;
          const truth = t[i][idx];
          const prob = p[i][idx];
          const pred = prob > 0.5 ? 1 : 0;
          preds.push({ truth, pred, correct: truth === pred });
          if (truth === pred) correct++;
          total++;
        }
      }
      stockAccuracies[sym] = total ? (correct / total) : 0;
      stockPredictions[sym] = preds;
    });

    return { stockAccuracies, stockPredictions };
  }
}
