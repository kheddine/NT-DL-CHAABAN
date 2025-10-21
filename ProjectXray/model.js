/**
 * model.js
 * TensorFlow.js model architecture and training/inference logic.
 * Includes CNN and optional MobileNetV2 transfer learning.
 */

let covidModel = null;

/** Build CNN model for chest X-ray classification */
async function buildBaseModel(transfer = false, learningRate = 0.001) {
  let model;
  if (transfer) {
    // Transfer learning using MobileNetV2 base
    const mobilenet = await tf.loadLayersModel(
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_140_224/classification/5/default/1',
      { fromTFHub: true }
    );
    const layer = mobilenet.getLayer('module_apply_default/MobilenetV2/Logits/AvgPool');
    const base = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
    base.trainable = false;

    model = tf.sequential();
    model.add(tf.layers.inputLayer({ inputShape: [224, 224, 3] }));
    model.add(base);
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
  } else {
    // Simple CNN for educational purposes
    model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape: [224, 224, 3], filters: 32, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
  }

  const optimizer = tf.train.adam(learningRate);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  covidModel = model;
  return model;
}

/** Train the model with given data */
async function trainModel(trainXs, trainYs, valXs, valYs, epochs = 5, batchSize = 8) {
  if (!covidModel) throw new Error('Model not initialized');

  const surface = { name: 'Training Metrics', tab: 'Training' };
  const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'acc'], { height: 250, callbacks: ['onEpochEnd'] });

  const history = await covidModel.fit(trainXs, trainYs, {
    epochs,
    batchSize,
    validationData: [valXs, valYs],
    shuffle: true,
    callbacks: fitCallbacks
  });

  return history;
}

/** Predict class probabilities for one or more preprocessed tensors */
async function predictImages(imageTensors) {
  if (!covidModel) throw new Error('Model not loaded');
  const preds = covidModel.predict(imageTensors);
  const probs = await preds.array();
  tf.dispose(preds);
  return probs;
}

/** Evaluate model on test data and compute metrics */
async function evaluateModel(testXs, testYs) {
  if (!covidModel) throw new Error('Model not trained');
  const evalResult = covidModel.evaluate(testXs, testYs);
  const [loss, acc] = await Promise.all(evalResult.map(t => t.data()));
  return { loss: loss[0], accuracy: acc[0] };
}

/** Dispose model and tensors to free memory */
function clearModel() {
  if (covidModel) {
    covidModel.dispose();
    covidModel = null;
  }
}
