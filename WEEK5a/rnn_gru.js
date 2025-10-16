// cnn_gru.js
export function createCNNGRUModel(stockCount) {
  const input = tf.input({shape:[12, 2*stockCount]});
  let x = tf.layers.conv1d({filters:64, kernelSize:3, padding:'same', activation:'relu'}).apply(input);
  x = tf.layers.dropout({rate:0.2}).apply(x);
  x = tf.layers.gru({units:64}).apply(x);
  x = tf.layers.dropout({rate:0.2}).apply(x);
  x = tf.layers.dense({units:64, activation:'relu'}).apply(x);
  const output = tf.layers.dense({units:3*stockCount, activation:'sigmoid'}).apply(x);
  const model = tf.model({inputs:input, outputs:output});
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['binaryAccuracy']
  });
  return model;
}

export async function trainModel(model, X_train, y_train, X_val, y_val, progressCb, statusCb) {
  const epochs = 50;
  const batchSize = 64;
  let bestValLoss = Infinity;
  let patience = 6;
  let noImprove = 0;
  let bestWeights = null;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const history = await model.fit(X_train, y_train, {
      epochs: 1,
      batchSize,
      validationData: [X_val, y_val],
      shuffle: true
    });
    const valLoss = history.history.loss[0];
    progressCb(epoch / epochs);
    statusCb(`Epoch ${epoch}/${epochs} - valLoss: ${valLoss.toFixed(4)}`);

    if (valLoss < bestValLoss) {
      bestValLoss = valLoss;
      bestWeights = model.getWeights().map(w => w.clone());
      noImprove = 0;
    } else {
      noImprove++;
      if (noImprove >= patience) {
        statusCb(`Early stopping at epoch ${epoch}`);
        if (bestWeights) model.setWeights(bestWeights);
        break;
      }
    }
    await tf.nextFrame();
  }
}

export async function evaluateModel(model, X_test, y_test, symbols) {
  const preds = model.predict(X_test);
  const predArr = await preds.array();
  const trueArr = await y_test.array();
  preds.dispose();
  const stockAccuracies = {};
  const stockPredictions = {};

  for (let s = 0; s < symbols.length; s++) {
    const sym = symbols[s];
    let correct = 0;
    const timeline = [];
    for (let i = 0; i < trueArr.length; i++) {
      for (let j = 0; j < 3; j++) {
        const truth = trueArr[i][s*3 + j] > 0.5 ? 1 : 0;
        const pred = predArr[i][s*3 + j] > 0.5 ? 1 : 0;
        const ok = truth === pred;
        if (ok) correct++;
        timeline.push({truth, pred, correct: ok});
      }
    }
    const acc = correct / (trueArr.length * 3);
    stockAccuracies[sym] = acc;
    stockPredictions[sym] = timeline;
  }

  return {stockAccuracies, stockPredictions};
}
