// rnn.js
// Defines and trains the RNN-based binary classification model.

export class StockRNN {
  constructor(inputShape, outputSize) {
    this.model = this.buildModel(inputShape, outputSize);
  }

  buildModel(inputShape, outputSize) {
    const model = tf.sequential();
    model.add(tf.layers.simpleRNN({ units: 64, returnSequences: false, inputShape }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: outputSize, activation: 'sigmoid' }));
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy'],
    });
    return model;
  }

  async train(X_train, y_train, epochs = 10, batchSize = 32, onEpochEnd = null) {
    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle: false,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpochEnd) onEpochEnd(epoch, logs);
          await tf.nextFrame();
        },
      },
    });
  }

  async evaluate(X_test, y_test) {
    const result = await this.model.evaluate(X_test, y_test);
    return { loss: result[0].dataSync()[0], accuracy: result[1].dataSync()[0] };
  }

  predict(X) {
    return this.model.predict(X);
  }
}
