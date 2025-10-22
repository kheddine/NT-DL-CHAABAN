import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

export class GRUModel {
    constructor(inputShape = [12, 20], outputSize = 30) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
        this.isTraining = false;
    }

    buildModel() {
        const model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.inputLayer({ inputShape: this.inputShape }));
        
        // 1D Convolution for feature extraction
        model.add(tf.layers.conv1d({
            filters: 16,  // Reduced for speed
            kernelSize: 2,
            activation: 'relu',
            padding: 'same'
        }));
        
        // First GRU layer
        model.add(tf.layers.gru({
            units: 32,  // Reduced for speed
            returnSequences: true,
            activation: 'tanh'
        }));
        
        // Second GRU layer
        model.add(tf.layers.gru({
            units: 16,  // Reduced for speed
            activation: 'tanh'
        }));
        
        // Dense layers
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        this.model = model;
        console.log('Model built successfully');
        return model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 30, batchSize = 16) {  // Reduced epochs and batch size
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        if (this.isTraining) {
            throw new Error('Model is already training');
        }

        this.isTraining = true;

        try {
            this.history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                verbose: 0,  // Reduce console output
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Clean up memory every few epochs
                        if (epoch % 5 === 0) {
                            await tf.nextFrame();
                        }
                        
                        console.log(`Epoch ${epoch + 1}/${epochs}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}, val_acc = ${logs.val_acc.toFixed(4)}`);
                        
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
                    },
                    onTrainEnd: () => {
                        this.isTraining = false;
                    }
                }
            });

            return this.history;
        } catch (error) {
            this.isTraining = false;
            throw error;
        }
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }
        
        // Ensure input is a tensor
        const inputTensor = X instanceof tf.Tensor ? X : tf.tensor(X);
        const prediction = this.model.predict(inputTensor);
        
        // Clean up if we created a new tensor
        if (inputTensor !== X) {
            inputTensor.dispose();
        }
        
        return prediction;
    }

    evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }
        return this.model.evaluate(X_test, y_test);
    }

    computePerStockAccuracy(predictions, y_true, symbols) {
        // Convert to arrays for processing
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
            
            accuracies[stock] = total > 0 ? correct / total : 0;
        });
        
        return accuracies;
    }

    async saveModel(name = 'gru-model') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        await this.model.save(`indexeddb://${name}`);
        console.log('Model saved successfully');
    }

    async loadModel(name = 'gru-model') {
        try {
            this.model = await tf.loadLayersModel(`indexeddb://${name}`);
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        this.isTraining = false;
    }
}
