import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/+esm';

export class GRUStockPredictor {
    constructor(inputShape = [12, 20], outputSize = 30) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
    }

    buildModel() {
        this.model = tf.sequential();
        
        // First GRU layer
        this.model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: this.inputShape
        }));
        
        // Second GRU layer
        this.model.add(tf.layers.gru({
            units: 32,
            returnSequences: false
        }));
        
        // Dropout for regularization
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer - 30 outputs (10 stocks × 3 days)
        this.model.add(tf.layers.dense({
            units: this.outputSize,
            activation: 'sigmoid'
        }));

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) {
            this.buildModel();
        }

        this.history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built or trained');
        }
        return this.model.predict(X);
    }

    evaluatePerStock(yTrue, yPred, symbols) {
        const stocks = symbols.length;
        const days = 3;
        
        // Convert tensors to arrays
        const trueArray = yTrue.arraySync();
        const predArray = yPred.arraySync();
        
        const results = {};
        symbols.forEach((symbol, index) => {
            results[symbol] = {
                accuracies: [],
                predictions: [],
                actuals: []
            };
        });

        // Calculate accuracy for each stock across all test samples
        for (let sample = 0; sample < trueArray.length; sample++) {
            for (let day = 0; day < days; day++) {
                for (let stock = 0; stock < stocks; stock++) {
                    const outputIndex = day * stocks + stock;
                    const actual = trueArray[sample][outputIndex];
                    const predicted = predArray[sample][outputIndex] > 0.5 ? 1 : 0;
                    const symbol = symbols[stock];
                    
                    results[symbol].predictions.push(predicted);
                    results[symbol].actuals.push(actual);
                }
            }
        }

        // Calculate final accuracies
        symbols.forEach(symbol => {
            const correct = results[symbol].predictions.reduce((sum, pred, idx) => {
                return sum + (pred === results[symbol].actuals[idx] ? 1 : 0);
            }, 0);
            results[symbol].accuracy = correct / results[symbol].predictions.length;
        });

        return results;
    }

    async saveModel(name = 'gru-stock-model') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        await this.model.save(`indexeddb://${name}`);
    }

    async loadModel(name = 'gru-stock-model') {
        this.model = await tf.loadLayersModel(`indexeddb://${name}`);
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
