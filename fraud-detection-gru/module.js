// FraudDetectionModel: GRU-based neural network for fraud detection
class FraudDetectionModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = [];
    }

    // Build GRU model architecture
    buildModel(inputShape) {
        this.model = tf.sequential({
            layers: [
                // GRU layer for sequence processing
                tf.layers.gru({
                    units: 32,
                    activation: 'tanh',
                    returnSequences: false,
                    inputShape: inputShape
                }),
                
                // Dropout for regularization
                tf.layers.dropout({ rate: 0.3 }),
                
                // Dense hidden layer
                tf.layers.dense({
                    units: 16,
                    activation: 'relu'
                }),
                
                // Output layer for binary classification
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });

        // Compile model with appropriate metrics
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy', 'precision', 'recall']
        });

        console.log('GRU Model built successfully');
        return this.model;
    }

    // Train the model with progress tracking
    async train(sequences, labels, epochs = 20, validationSplit = 0.2) {
        if (!this.model) {
            throw new Error('Model must be built before training');
        }

        this.trainingHistory = [];
        
        // Training configuration
        const trainingConfig = {
            epochs: epochs,
            validationSplit: validationSplit,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // Store training progress
                    this.trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        accuracy: logs.acc,
                        valLoss: logs.val_loss,
                        valAccuracy: logs.val_acc
                    });
                    
                    // Dispatch custom event for UI updates
                    const event = new CustomEvent('trainingProgress', {
                        detail: {
                            epoch: epoch + 1,
                            loss: logs.loss.toFixed(4),
                            accuracy: (logs.acc * 100).toFixed(2),
                            valLoss: logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A',
                            valAccuracy: logs.val_acc ? (logs.val_acc * 100).toFixed(2) : 'N/A'
                        }
                    });
                    document.dispatchEvent(event);
                }
            }
        };

        try {
            await this.model.fit(sequences, labels, trainingConfig);
            this.isTrained = true;
            console.log('Model training completed');
        } catch (error) {
            console.error('Training failed:', error);
            throw error;
        }
    }

    // Evaluate model on test data
    async evaluate(testSequences, testLabels) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before evaluation');
        }

        const evaluation = this.model.evaluate(testSequences, testLabels);
        const loss = await evaluation[0].data();
        const accuracy = await evaluation[1].data();
        
        // Clean up tensors
        tf.dispose(evaluation);

        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    // Make predictions
    async predict(sequences) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before prediction');
        }

        const predictions = this.model.predict(sequences);
        const predictionData = await predictions.data();
        
        // Clean up
        tf.dispose(predictions);

        return Array.from(predictionData);
    }

    // Get model summary for display
    getModelSummary() {
        if (!this.model) return 'Model not built yet';
        
        let summary = 'Model Architecture:\n';
        this.model.layers.forEach((layer, index) => {
            summary += `${index + 1}. ${layer.getClassName()}: ${JSON.stringify(layer.outputShape)}\n`;
        });
        
        return summary;
    }

    // Clean up model and tensors
    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default FraudDetectionModel;
