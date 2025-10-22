// module.js
// Defines GRU model using TensorFlow.js
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.16.0/dist/tf.min.js";

export class GRUFraudModel {
  constructor(inputShape) {
    this.model = tf.sequential();
    this.model.add(tf.layers.gru({ units: 32, inputShape, returnSequences: false }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 16, activation: "relu" }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"]
    });
  }

  async train(X_train, y_train, X_val, y_val, onEpoch) {
    const h = await this.model.fit(X_train, y_train, {
      validationData: [X_val, y_val],
      epochs: 10,
      batchSize: 128,
      callbacks: {
        onEpochEnd: (epoch, logs) => onEpoch?.(epoch, logs)
      }
    });
    return h;
  }

  async evaluate(X_test, y_test) {
    const evalRes = await this.model.evaluate(X_test, y_test);
    const loss = evalRes[0].dataSync()[0];
    const acc = evalRes[1].dataSync()[0];
    return { loss, acc };
  }

  predict(X) {
    return this.model.predict(X);
  }
}
