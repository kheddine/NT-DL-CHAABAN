// FraudDetectionModel: GRU-based neural network for fraud detection
class FraudDetectionModel {
    constructor(processCallback) {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = [];
        this.processCallback = processCallback;
    }

    // Update process window
    updateProcess(message, type = 'info', isActive = false) {
        if (this.processCallback) {
            this.processCallback(message, type, isActive);
        }
    }

    // Build GRU model architecture
    buildModel(inputShape) {
        this.updateProcess('Building GRU model architecture...', 'info', true);
        
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

        this.updateProcess('Model architecture created, compiling...', 'info', true);

        // Compile model with appropriate metrics
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy', 'precision', 'recall']
        });

        this.updateProcess('GRU Model built and compiled successfully', 'success');
        console.log('GRU Model built successfully');
        return this.model;
    }

    // Train the model with progress tracking
    async train(sequences, labels, epochs = 20, validationSplit = 0.2) {
        if (!this.model) {
            this.updateProcess('Error: Model must be built before training', 'error');
            throw new Error('Model must be built before training');
        }

        this.updateProcess(`Starting training for ${epochs} epochs...`, 'info', true);
        this.trainingHistory = [];
        
        // Training configuration
        const trainingConfig = {
            epochs: epochs,
            validationSplit: validationSplit,
            callbacks: {
                onEpochBegin: (epoch) => {
                    this.updateProcess(`Starting epoch ${epoch + 1}/${epochs}...`, 'info', true);
                },
                onEpochEnd: (epoch, logs) => {
                    // Store training progress
                    this.trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        accuracy: logs.acc,
                        valLoss: logs.val_loss,
                        valAccuracy: logs.val_acc
                    });
                    
                    const progressMessage = `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Acc: ${(logs.acc * 100).toFixed(2)}%`;
                    this.updateProcess(progressMessage, 'info');
                    
                    // Update progress percentage
                    const progressPercent = ((epoch + 1) / epochs) * 100;
                    
                    // Dispatch custom event for UI updates
                    const event = new CustomEvent('trainingProgress', {
                        detail: {
                            epoch: epoch + 1,
                            loss: logs.loss.toFixed(4),
                            accuracy: (logs.acc * 100).toFixed(2),
                            valLoss: logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A',
                            valAccuracy: logs.val_acc ? (logs.val_acc * 100).toFixed(2) : 'N/A',
                            progressPercent: progressPercent
                        }
                    });
                    document.dispatchEvent(event);
                },
                onTrainBegin: () => {
                    this.updateProcess('Training process started...', 'info', true);
                },
                onTrainEnd: () => {
                    this.updateProcess('Training process completed!', 'success');
                }
            }
        };

        try {
            await this.model.fit(sequences, labels, trainingConfig);
            this.isTrained = true;
            this.updateProcess('Model training completed successfully!', 'success');
            console.log('Model training completed');
        } catch (error) {
            this.updateProcess(`Training failed: ${error.message}`, 'error');
            console.error('Training failed:', error);
            throw error;
        }
    }

    // Evaluate model on test data
    async evaluate(testSequences, testLabels) {
        if (!this.isTrained) {
            this.updateProcess('Error: Model must be trained before evaluation', 'error');
            throw new Error('Model must be trained before evaluation');
        }

        this.updateProcess('Starting model evaluation...', 'info', true);

        const evaluation = this.model.evaluate(testSequences, testLabels);
        const loss = await evaluation[0].data();
        const accuracy = await evaluation[1].data();
        
        // Clean up tensors
        tf.dispose(evaluation);

        this.updateProcess(`Evaluation completed - Loss: ${loss[0].toFixed(4)}, Accuracy: ${(accuracy[0] * 100).toFixed(2)}%`, 'success');

        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    // Make predictions
    async predict(sequences) {
        if (!this.isTrained) {
            this.updateProcess('Error: Model must be trained before prediction', 'error');
            throw new Error('Model must be trained before prediction');
        }

        this.updateProcess('Making predictions...', 'info', true);

        const predictions = this.model.predict(sequences);
        const predictionData = await predictions.data();
        
        // Clean up
        tf.dispose(predictions);

        this.updateProcess(`Predictions completed for ${predictionData.length} samples`, 'success');

        return Array.from(predictionData);
    }

    // Get model summary for display
    getModelSummary() {
        if (!this.model) {
            this.updateProcess('Model not built yet', 'warning');
            return 'Model not built yet';
        }
        
        let summary = 'Model Architecture:\n';
        this.model.layers.forEach((layer, index) => {
            summary += `${index + 1}. ${layer.getClassName()}: ${JSON.stringify(layer.outputShape)}\n`;
        });
        
        this.updateProcess('Model summary generated', 'info');
        return summary;
    }

    // Clean up model and tensors
    dispose() {
        this.updateProcess('Cleaning up model and tensors...', 'info');
        if (this.model) {
            this.model.dispose();
            this.updateProcess('Model cleanup completed', 'success');
        }
    }
}

export default FraudDetectionModel;
