// rnn_gru.js
class RNN_GRU_Model {
    constructor(inputShape, outputSize) {
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.model = null;
        this.history = [];
        this.isTraining = false;
    }

    // Create the hybrid RNN + GRU model
    createModel() {
        const model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.simpleRNN({
            units: 64,
            returnSequences: true,
            inputShape: this.inputShape
        }));
        
        // First dropout for regularization
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // GRU layer
        model.add(tf.layers.gru({
            units: 64
        }));
        
        // Second dropout
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Dense layers
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        // Output layer - sigmoid for binary classification
        model.add(tf.layers.dense({
            units: this.outputSize,
            activation: 'sigmoid'
        }));
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });
        
        this.model = model;
        console.log('Model created successfully');
        model.summary();
        
        return model;
    }

    // Train the model with early stopping
    async train(X_train, y_train, X_val, y_val, callbacks = {}) {
        if (!this.model) {
            this.createModel();
        }

        this.isTraining = true;
        this.history = [];
        
        const epochs = 50;
        const batchSize = 64;
        let bestLoss = Infinity;
        let patience = 6;
        let patienceCounter = 0;
        let bestWeights = null;
        
        const { onEpochEnd, onTrainEnd } = callbacks;
        
        try {
            // Initialize WebGL backend explicitly
            await tf.setBackend('webgl');
            console.log('Using WebGL backend for training');
        } catch (error) {
            console.warn('WebGL not available, falling back to CPU');
            await tf.setBackend('cpu');
        }
        
        for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
            // Train for one epoch
            const history = await this.model.fit(X_train, y_train, {
                epochs: 1,
                batchSize: batchSize,
                validationData: X_val && y_val ? [X_val, y_val] : null,
                shuffle: true,
                verbose: 0
            });
            
            const epochHistory = history.history;
            const trainLoss = epochHistory.loss[0];
            const trainAcc = epochHistory.binaryAccuracy[0];
            const valLoss = epochHistory.val_loss ? epochHistory.val_loss[0] : null;
            const valAcc = epochHistory.val_binaryAccuracy ? epochHistory.val_binaryAccuracy[0] : null;
            
            this.history.push({
                epoch: epoch + 1,
                loss: trainLoss,
                accuracy: trainAcc,
                val_loss: valLoss,
                val_accuracy: valAcc
            });
            
            // Early stopping logic
            const currentLoss = valLoss !== null ? valLoss : trainLoss;
            
            if (currentLoss < bestLoss) {
                bestLoss = currentLoss;
                patienceCounter = 0;
                bestWeights = this.model.getWeights();
            } else {
                patienceCounter++;
            }
            
            if (patienceCounter >= patience) {
                console.log(`Early stopping at epoch ${epoch + 1}`);
                if (bestWeights) {
                    this.model.setWeights(bestWeights);
                }
                break;
            }
            
            // Call progress callback
            if (onEpochEnd) {
                const progress = ((epoch + 1) / epochs) * 100;
                onEpochEnd({
                    epoch: epoch + 1,
                    progress,
                    loss: trainLoss,
                    accuracy: trainAcc,
                    valLoss,
                    valAccuracy: valAcc
                });
            }
            
            // Force garbage collection
            if (epoch % 5 === 0) {
                tf.engine().startScope();
                tf.engine().endScope();
            }
        }
        
        this.isTraining = false;
        
        if (onTrainEnd) {
            onTrainEnd(this.history);
        }
        
        return this.history;
    }

    // Stop training
    stopTraining() {
        this.isTraining = false;
    }

    // Make predictions
    predict(X) {
        if (!this.model) {
            throw new Error('Model not trained yet');
        }
        return this.model.predict(X);
    }

    // Evaluate model and return per-stock metrics
    async evaluate(X_test, y_test, symbols) {
        if (!this.model) {
            throw new Error('Model not trained yet');
        }
        
        const predictions = this.predict(X_test);
        const yPred = predictions.arraySync();
        const yTrue = y_test.arraySync();
        
        // Calculate per-stock accuracy
        const stockAccuracies = {};
        const stockPredictions = {};
        
        symbols.forEach((symbol, symbolIndex) => {
            let correct = 0;
            let total = 0;
            const predictions = [];
            
            for (let i = 0; i < yTrue.length; i++) {
                const trueVal = yTrue[i][symbolIndex];
                const predVal = yPred[i][symbolIndex] > 0.5 ? 1 : 0;
                const isCorrect = trueVal === predVal ? 1 : 0;
                
                if (i < 50) { // Limit to first 50 for timeline
                    predictions.push({
                        truth: trueVal,
                        pred: predVal,
                        correct: isCorrect
                    });
                }
                
                if (trueVal !== undefined) {
                    correct += isCorrect;
                    total++;
                }
            }
            
            const accuracy = total > 0 ? (correct / total) : 0;
            stockAccuracies[symbol] = accuracy;
            stockPredictions[symbol] = predictions;
        });
        
        // Clean up tensors
        predictions.dispose();
        
        return {
            stockAccuracies,
            stockPredictions
        };
    }

    // Save model weights (stretch feature)
    async saveModel() {
        if (!this.model) return null;
        
        try {
            const saveResult = await this.model.save('indexeddb://stock-prediction-model');
            console.log('Model saved to IndexedDB');
            return saveResult;
        } catch (error) {
            console.warn('Failed to save model:', error);
            return null;
        }
    }

    // Load model weights (stretch feature)
    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://stock-prediction-model');
            console.log('Model loaded from IndexedDB');
            return true;
        } catch (error) {
            console.warn('No saved model found:', error);
            return false;
        }
    }
}

// Export for use in other modules
window.RNN_GRU_Model = RNN_GRU_Model;
