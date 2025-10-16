// rnn_gru.js
// Hybrid SimpleRNN + GRU with early stopping and robust training start.

export class RNN_GRU_Model {
  constructor(inputShape, outputSize) {
    this.inputShape = inputShape;   // [30, 4*stocks]
    this.outputSize = outputSize;   // 3 * stocks
    this.model = null;
  }

  async ensureBackend() {
    // Try WebGL; if it fails, fall back to CPU so training still starts.
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      console.log('TF backend:', tf.getBackend());
    } catch (e) {
      console.warn('WebGL unavailable, falling back to CPU.', e);
      await tf.setBackend('cpu');
      await tf.ready();
    }
  }

  buildModel() {
    this.model = tf.sequential({
      layers: [
        tf.layers.simpleRNN({ units: 64, returnSequences: true, inputShape: this.inputShape }),
        tf.layers.dropout({ rate: 0.2 }),
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
    console.log('Train shapes:', Xtr.shape, ytr.shape);
    console.log('Val shapes:', Xv.shape, yv.shape);

    await this.ensureBackend();
    if (!this.model) this.buildModel();

    const hasVal = (Xv?.shape?.[0] || 0) > 0 && (yv?.shape?.[0] || 0) > 0;
    const valData = hasVal ? [Xv, yv] : null;
    if (!hasVal) console.warn('⚠️ No validation split; training without val.');

    const earlyStop = tf.callbacks.earlyStopping({
      monitor: hasVal ? 'val_loss' : 'loss',
      patience: 6,
      restoreBestWeights: true
    });

    const cb = {
      onEpochBegin: async (epoch) => {
        console.log(`Epoch ${epoch + 1}/${epochs} start`);
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        const p = document.getElementById('trainingProgress');
        const s = document.getElementById('status');
        if (p) p.value = ((epoch + 1) / epochs) * 100;
        if (s) {
          const base = `Epoch ${epoch + 1}/${epochs} | loss: ${logs.loss.toFixed(4)} | acc: ${(logs.binaryAccuracy * 100).toFixed(1)}%`;
          s.textContent = hasVal && logs.val_binaryAccuracy != null
            ? `${base} | val_acc: ${(logs.val_binaryAccuracy * 100).toFixed(1)}%`
            : base;
        }
        await tf.nextFrame();
      }
    };

    console.log('🚀 Starting model.fit...');
    const hist = await this.model.fit(Xtr, ytr, {
      epochs,
      batchSize,
      shuffle: true,
      validationData: valData,
      callbacks: [earlyStop, cb]
    });
    console.log('✅ Training finished');
    return hist;
  }

  predict(X) {
    const out = this.model.predict(X);
    // tfjs predict may return a Tensor or array of Tensors depending on model; normalize to Tensor.
    if (Array.isArray(out)) return out[0];
    return out;
  }

  evaluate(yTrue, yPred, symbols, horizon = 3) {
    const t = yTrue.arraySync();
    const p = yPred.arraySync();
    const acc = {};
    const detail = {};

    symbols.forEach((sym, si) => {
      let correct = 0, total = 0;
      const preds = [];
      for (let i = 0; i < t.length; i++) {
        for (let h = 0; h < horizon; h++) {
          const idx = si * horizon + h;
          const truth = t[i][idx];
          const pred  = p[i][idx] > 0.5 ? 1 : 0;
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
