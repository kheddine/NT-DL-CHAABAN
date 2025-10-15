import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class RNNModel {
    constructor() {
        this.model = null;
        this.history = null;
    }

    buildModel(inputShape = [30, 40], outputUnits = 30) {
        this.model = tf.sequential();
        
        // First RNN layer
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            inputShape: inputShape
        }));
        
        // Second RNN layer
        this.model.add(tf.layers.lstm({
            units: 32,
            returnSequences: false
        }));
        
        // Dense layer before output
        this.model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        // Output layer - 30 binary classifications
        this.model.add(tf.layers.dense({
            units: outputUnits,
            activation: 'sigmoid'
        }));
        
        // Compile model
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });
        
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) {
            throw new Error('Model must be built before training');
        }

        this.history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                    // Update UI if callback provided
                    if (this.onEpochEnd) {
                        this.onEpochEnd(epoch, logs);
                    }
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model must be built before prediction');
        }
        return this.model.predict(X);
    }

    evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model must be built before evaluation');
        }
        return this.model.evaluate(X_test, y_test);
    }

    computeStockAccuracies(predictions, y_test, symbols) {
        const predArray = predictions.arraySync();
        const testArray = y_test.arraySync();
        
        const stockAccuracies = {};
        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            
            for (let sampleIdx = 0; sampleIdx < predArray.length; sampleIdx++) {
                for (let dayOffset = 0; dayOffset < 3; dayOffset++) {
                    const outputIdx = stockIdx + dayOffset * symbols.length;
                    const pred = predArray[sampleIdx][outputIdx] > 0.5 ? 1 : 0;
                    const actual = testArray[sampleIdx][outputIdx];
                    
                    if (actual !== undefined) {
                        if (pred === actual) correct++;
                        total++;
                    }
                }
            }
            
            stockAccuracies[symbol] = total > 0 ? correct / total : 0;
        });
        
        return stockAccuracies;
    }

    async saveModel(name = 'rnn-model') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        await this.model.save(`indexeddb://${name}`);
    }

    async loadModel(name = 'rnn-model') {
        this.model = await tf.loadLayersModel(`indexeddb://${name}`);
        return this.model;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
