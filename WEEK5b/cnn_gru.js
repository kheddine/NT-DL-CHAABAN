import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

export class GRUModel {
    constructor(inputShape = [12, 20], outputSize = 30) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
    }

    buildModel() {
        const model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.inputLayer({ inputShape: this.inputShape }));
        
        // 1D Convolution for feature extraction
        model.add(tf.layers.conv1d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        // First GRU layer
        model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            activation: 'tanh'
        }));
        
        // Second GRU layer
        model.add(tf.layers.gru({
            units: 32,
            activation: 'tanh'
        }));
        
        // Dense layers
        model.add(tf.layers.dense({ units: 50, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.3 }));
        model.add(tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        this.model = model;
        return model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}, val_accuracy = ${logs.val_acc.toFixed(4)}`);
                    
                    // Dispatch custom event for UI updates
                    const event = new CustomEvent('trainingProgress', {
                        detail: {
                            epoch: epoch + 1,
                            epochs: epochs,
                            loss: logs.loss,
                            accuracy: logs.acc,
                            valLoss: logs.val_loss,
                            valAccuracy: logs.val_acc
                        }
                    });
                    window.dispatchEvent(event);
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }
        return this.model.predict(X);
    }

    evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }
        return this.model.evaluate(X_test, y_test);
    }

    computePerStockAccuracy(predictions, y_true, symbols) {
        const predData = predictions.arraySync();
        const trueData = y_true.arraySync();
        const stocks = symbols;
        const days = 3;
        
        const accuracies = {};
        
        stocks.forEach((stock, stockIdx) => {
            let correct = 0;
            let total = 0;
            
            for (let sampleIdx = 0; sampleIdx < predData.length; sampleIdx++) {
                for (let day = 0; day < days; day++) {
                    const predIdx = stockIdx * days + day;
                    const predicted = predData[sampleIdx][predIdx] > 0.5 ? 1 : 0;
                    const actual = trueData[sampleIdx][predIdx];
                    
                    if (predicted === actual) {
                        correct++;
                    }
                    total++;
                }
            }
            
            accuracies[stock] = correct / total;
        });
        
        return accuracies;
    }

    async saveModel(name = 'gru-model') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        await this.model.save(`indexeddb://${name}`);
    }

    async loadModel(name = 'gru-model') {
        this.model = await tf.loadLayersModel(`indexeddb://${name}`);
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
