/**
 * Pneumonia Chest X-ray Classification Model
 * Optimized CNN for browser-based medical image analysis
 * Educational purpose only - NOT for clinical use
 */

class PneumoniaClassifier {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        this.classNames = ['Normal', 'Pneumonia'];
        this.inputSize = [150, 150]; // Reduced for better performance
    }

    /**
     * Initialize TensorFlow.js with fallback backends
     */
    async initializeTFJS() {
        const backends = ['webgl', 'cpu'];
        let backendInfo = '';
        
        for (const backend of backends) {
            try {
                await tf.setBackend(backend);
                backendInfo = `Using backend: ${backend}`;
                console.log(`Successfully initialized TensorFlow.js with ${backend} backend`);
                break;
            } catch (error) {
                console.warn(`Failed to initialize ${backend} backend:`, error);
                continue;
            }
        }
        
        if (!tf.getBackend()) {
            throw new Error('Failed to initialize TensorFlow.js with any backend');
        }
        
        return backendInfo;
    }

    /**
     * Create optimized CNN model for pneumonia classification
     */
    createModel() {
        const model = tf.sequential();
        
        // First Convolutional Block
        model.add(tf.layers.conv2d({
            inputShape: [this.inputSize[0], this.inputSize[1], 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv1'
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool1'
        }));
        
        // Second Convolutional Block
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv2'
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool2'
        }));
        
        // Third Convolutional Block
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv3'
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool3'
        }));
        
        // Global Average Pooling instead of Flatten to reduce parameters
        model.add(tf.layers.globalAveragePooling2d());
        
        // Classification Head
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'dense1'
        }));
        model.add(tf.layers.dropout({
            rate: 0.5,
            name: 'dropout'
        }));
        
        // Output layer: 2 classes (Normal, Pneumonia)
        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax',
            name: 'output'
        }));
        
        // Compile model with optimized settings
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Optimized CNN Model created successfully');
        console.log('Model summary:');
        model.summary();
        
        return model;
    }

    /**
     * Train the model with medical imaging data
     */
    async trainModel(trainingData, validationData, config, onProgress = null) {
        if (!this.model) {
            throw new Error('Model not initialized. Call createModel() first.');
        }

        this.isTraining = true;
        this.trainingHistory = { loss: [], accuracy: [], valLoss: [], valAccuracy: [] };

        const { epochs, batchSize, validationSplit } = config;
        
        try {
            const history = await this.model.fit(trainingData.xs, trainingData.ys, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: validationSplit,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Update training history
                        this.trainingHistory.loss.push(logs.loss);
                        this.trainingHistory.accuracy.push(logs.acc);
                        this.trainingHistory.valLoss.push(logs.val_loss);
                        this.trainingHistory.valAccuracy.push(logs.val_acc);
                        
                        // Update progress callback
                        if (onProgress) {
                            const progress = {
                                epoch: epoch + 1,
                                totalEpochs: epochs,
                                loss: logs.loss,
                                accuracy: logs.acc,
                                valLoss: logs.val_loss,
                                valAccuracy: logs.val_acc
                            };
                            onProgress(progress);
                        }
                        
                        // Update visualization
                        this.updateTrainingCharts();
                        
                        // Memory cleanup
                        await tf.nextFrame();
                    },
                    onTrainEnd: () => {
                        this.isTraining = false;
                        console.log('Training completed');
                    }
                }
            });
            
            return history;
        } catch (error) {
            this.isTraining = false;
            console.error('Training error:', error);
            throw error;
        }
    }

    /**
     * Predict pneumonia probability for single image
     */
    async predict(imageTensor) {
        if (!this.model) {
            throw new Error('Model not loaded. Please train or load a model first.');
        }

        try {
            // Ensure tensor has correct shape [1, 150, 150, 3]
            const batchedImage = imageTensor.expandDims(0);
            
            // Make prediction
            const prediction = this.model.predict(batchedImage);
            const probabilities = await prediction.data();
            
            // Clean up tensors
            batchedImage.dispose();
            prediction.dispose();
            
            return {
                pneumonia: probabilities[1],  // Pneumonia probability
                normal: probabilities[0],     // Normal probability
                predictedClass: probabilities[1] > probabilities[0] ? 'Pneumonia' : 'Normal',
                confidence: Math.max(probabilities[0], probabilities[1])
            };
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    /**
     * Batch prediction for multiple images
     */
    async predictBatch(imageTensors) {
        const predictions = [];
        
        for (let i = 0; i < imageTensors.length; i++) {
            try {
                const prediction = await this.predict(imageTensors[i]);
                predictions.push(prediction);
                
                // Periodic cleanup to prevent memory issues
                if (i % 10 === 0) {
                    await tf.nextFrame();
                }
            } catch (error) {
                console.error(`Batch prediction error for image ${i}:`, error);
                predictions.push({
                    pneumonia: 0.5,
                    normal: 0.5,
                    predictedClass: 'Unknown',
                    confidence: 0.5,
                    error: true
                });
            }
        }
        
        return predictions;
    }

    /**
     * Generate Grad-CAM heatmap for model interpretability
     */
    async generateHeatmap(imageTensor) {
        if (!this.model) return null;

        try {
            // Get the last convolutional layer
            const convLayer = this.model.getLayer('conv3');
            if (!convLayer) return null;

            // Create a model that outputs both conv layer and final prediction
            const gradModel = tf.model({
                inputs: this.model.inputs,
                outputs: [convLayer.output, this.model.output]
            });

            const [convOutputs, predictions] = tf.tidy(() => {
                const batchedImage = imageTensor.expandDims(0);
                return gradModel.predict(batchedImage);
            });

            // Get pneumonia class output
            const output = predictions.gather([1], 1); // Class 1: Pneumonia
            
            // Compute gradients
            const grads = tf.grad(() => output).grad(convOutputs);
            
            // Global average pooling of gradients
            const pooledGrads = tf.mean(grads, [0, 1, 2]);
            
            // Weight the convolutional outputs with gradients
            const heatmap = convOutputs.mul(pooledGrads).mean(2).squeeze();
            
            // Normalize heatmap
            const min = heatmap.min();
            const max = heatmap.max();
            const normalizedHeatmap = heatmap.sub(min).div(max.sub(min));
            
            // Resize to original image size
            const resizedHeatmap = tf.image.resizeBilinear(
                normalizedHeatmap.expandDims(2), 
                [this.inputSize[0], this.inputSize[1]]
            ).squeeze();
            
            const heatmapData = await resizedHeatmap.data();
            
            // Cleanup
            tf.dispose([convOutputs, predictions, grads, pooledGrads, heatmap, normalizedHeatmap, resizedHeatmap]);
            gradModel.dispose();
            
            return heatmapData;
        } catch (error) {
            console.error('Heatmap generation error:', error);
            return null;
        }
    }

    /**
     * Calculate model performance metrics
     */
    calculateMetrics(predictions, trueLabels) {
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
        predictions.forEach((pred, idx) => {
            if (pred.error) return; // Skip failed predictions
            
            const trueLabel = trueLabels[idx];
            const predLabel = pred.predictedClass;
            
            if (trueLabel === 'Pneumonia' && predLabel === 'Pneumonia') tp++;
            else if (trueLabel === 'Normal' && predLabel === 'Normal') tn++;
            else if (trueLabel === 'Normal' && predLabel === 'Pneumonia') fp++;
            else if (trueLabel === 'Pneumonia' && predLabel === 'Normal') fn++;
        });
        
        const total = tp + tn + fp + fn;
        if (total === 0) {
            return {
                accuracy: 0,
                precision: 0,
                recall: 0,
                f1Score: 0,
                confusionMatrix: { tp, tn, fp, fn }
            };
        }
        
        const accuracy = (tp + tn) / total;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
        
        return {
            accuracy: Math.round(accuracy * 10000) / 100,
            precision: Math.round(precision * 10000) / 100,
            recall: Math.round(recall * 10000) / 100,
            f1Score: Math.round(f1Score * 10000) / 100,
            confusionMatrix: { tp, tn, fp, fn }
        };
    }

    /**
     * Update training progress visualization
     */
    updateTrainingCharts() {
        if (this.trainingHistory.loss.length === 0) return;
        
        const lossData = {
            values: {
                'Training Loss': this.trainingHistory.loss,
                'Validation Loss': this.trainingHistory.valLoss
            },
            x: Array.from({length: this.trainingHistory.loss.length}, (_, i) => i + 1)
        };
        
        const accuracyData = {
            values: {
                'Training Accuracy': this.trainingHistory.accuracy,
                'Validation Accuracy': this.trainingHistory.valAccuracy
            },
            x: Array.from({length: this.trainingHistory.accuracy.length}, (_, i) => i + 1)
        };
        
        // Update tfjs-vis charts
        const vizContainer = document.getElementById('modelVisualization');
        if (vizContainer) {
            try {
                tfvis.show.history(vizContainer, lossData, ['loss'], {
                    xLabel: 'Epoch',
                    yLabel: 'Loss',
                    height: 300
                });
                
                tfvis.show.history(vizContainer, accuracyData, ['acc'], {
                    xLabel: 'Epoch',
                    yLabel: 'Accuracy',
                    height: 300
                });
            } catch (error) {
                console.warn('Visualization update failed:', error);
            }
        }
    }

    /**
     * Save model to browser storage
     */
    async saveModel(modelName = 'pneumonia-classifier') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        try {
            await this.model.save(`indexeddb://${modelName}`);
            console.log('Model saved successfully');
            return true;
        } catch (error) {
            console.error('Error saving model:', error);
            throw error;
        }
    }

    /**
     * Load model from browser storage
     */
    async loadModel(modelName = 'pneumonia-classifier') {
        try {
            this.model = await tf.loadLayersModel(`indexeddb://${modelName}`);
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    /**
     * Check if a model exists in storage
     */
    async modelExists(modelName = 'pneumonia-classifier') {
        try {
            // Try to load model info to check if it exists
            const models = await tf.io.listModels();
            return models.hasOwnProperty(`indexeddb://${modelName}`);
        } catch (error) {
            return false;
        }
    }

    /**
     * Dispose model to free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }

    /**
     * Get memory usage statistics
     */
    getMemoryStats() {
        return tf.memory();
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PneumoniaClassifier;
}
