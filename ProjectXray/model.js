/**
 * COVID-19 Chest X-ray Classification Model
 * TensorFlow.js implementation for browser-based medical image analysis
 * FOR EDUCATIONAL PURPOSES ONLY
 */

class COVID19Classifier {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.classNames = ['COVID-19', 'Normal'];
        this.inputShape = [224, 224, 3];
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
    }

    /**
     * Create CNN model for chest X-ray classification
     * Architecture optimized for medical image analysis
     */
    createCNNModel() {
        console.log('Creating CNN model for COVID-19 classification...');
        
        const model = tf.sequential({
            layers: [
                // Input layer
                tf.layers.conv2d({
                    inputShape: this.inputShape,
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    name: 'conv1'
                }),
                tf.layers.maxPooling2d({
                    poolSize: 2,
                    name: 'pool1'
                }),
                
                // Second convolutional block
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    name: 'conv2'
                }),
                tf.layers.maxPooling2d({
                    poolSize: 2,
                    name: 'pool2'
                }),
                
                // Third convolutional block
                tf.layers.conv2d({
                    filters: 128,
                    kernelSize: 3,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    name: 'conv3'
                }),
                tf.layers.maxPooling2d({
                    poolSize: 2,
                    name: 'pool3'
                }),
                
                // Fourth convolutional block for deeper feature extraction
                tf.layers.conv2d({
                    filters: 256,
                    kernelSize: 3,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    name: 'conv4'
                }),
                tf.layers.maxPooling2d({
                    poolSize: 2,
                    name: 'pool4'
                }),
                
                // Flatten and dense layers
                tf.layers.flatten({name: 'flatten'}),
                
                tf.layers.dense({
                    units: 512,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    name: 'dense1'
                }),
                tf.layers.dropout({
                    rate: 0.5,
                    name: 'dropout1'
                }),
                
                tf.layers.dense({
                    units: 128,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    name: 'dense2'
                }),
                tf.layers.dropout({
                    rate: 0.3,
                    name: 'dropout2'
                }),
                
                // Output layer
                tf.layers.dense({
                    units: 2,
                    activation: 'softmax',
                    name: 'output'
                })
            ]
        });

        // Compile model with medical imaging-appropriate parameters
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log('CNN model created successfully');
        this.model = model;
        return model;
    }

    /**
     * Create transfer learning model using MobileNetV2
     * Better feature extraction for medical images
     */
    async createTransferLearningModel() {
        console.log('Creating transfer learning model...');
        
        // Load MobileNetV2 base
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json'
        );

        // Freeze base model layers
        mobilenet.layers.forEach(layer => {
            layer.trainable = false;
        });

        // Create new model on top
        const model = tf.sequential({
            layers: [
                // MobileNetV2 base
                mobilenet,
                
                // Custom classifier head for medical images
                tf.layers.globalAveragePooling2d(),
                tf.layers.dropout({rate: 0.5}),
                tf.layers.dense({
                    units: 128,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({rate: 0.3}),
                tf.layers.dense({
                    units: 2,
                    activation: 'softmax'
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log('Transfer learning model created successfully');
        this.model = model;
        return model;
    }

    /**
     * Train model with medical imaging considerations
     */
    async trainModel(trainingData, validationData, config = {}) {
        if (this.isTraining) {
            throw new Error('Model is already training');
        }

        const {
            epochs = 10,
            batchSize = 8,
            learningRate = 0.001,
            callbacks = []
        } = config;

        // Update optimizer learning rate
        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        this.isTraining = true;
        this.trainingHistory = { loss: [], accuracy: [], valLoss: [], valAccuracy: [] };

        console.log('Starting model training...');

        try {
            const history = await this.model.fit(trainingData.xs, trainingData.ys, {
                epochs,
                batchSize,
                validationData: validationData ? [validationData.xs, validationData.ys] : null,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        // Store training history
                        this.trainingHistory.loss.push(logs.loss);
                        this.trainingHistory.accuracy.push(logs.acc);
                        this.trainingHistory.valLoss.push(logs.val_loss);
                        this.trainingHistory.valAccuracy.push(logs.val_acc);

                        // Update UI
                        if (callbacks.onEpochEnd) {
                            callbacks.onEpochEnd(epoch, logs);
                        }

                        // Memory management
                        tf.tidy(() => {
                            // Cleanup between epochs
                        });
                    },
                    onTrainEnd: () => {
                        this.isTraining = false;
                        if (callbacks.onTrainEnd) {
                            callbacks.onTrainEnd();
                        }
                    }
                },
                yieldEvery: 'epoch'
            });

            return history;
        } catch (error) {
            this.isTraining = false;
            console.error('Training error:', error);
            throw error;
        }
    }

    /**
     * Predict COVID-19 probability for single image
     * Includes confidence calibration for medical context
     */
    async predict(imageTensor) {
        if (!this.model) {
            throw new Error('Model not initialized');
        }

        try {
            // Ensure tensor is properly shaped
            const processedTensor = imageTensor.expandDims(0);
            
            // Make prediction
            const prediction = this.model.predict(processedTensor);
            const probabilities = await prediction.data();
            
            // Clean up tensors
            processedTensor.dispose();
            prediction.dispose();

            // Format results with medical confidence levels
            const covidProb = probabilities[1]; // COVID-19 probability
            const normalProb = probabilities[0]; // Normal probability
            
            return {
                className: covidProb > normalProb ? 'COVID-19' : 'Normal',
                confidence: Math.max(covidProb, normalProb),
                probabilities: {
                    'COVID-19': covidProb,
                    'Normal': normalProb
                },
                confidenceLevel: this.getConfidenceLevel(Math.max(covidProb, normalProb))
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
        if (!this.model) {
            throw new Error('Model not initialized');
        }

        try {
            const batchTensor = tf.stack(imageTensors);
            const predictions = this.model.predict(batchTensor);
            const results = await predictions.data();
            
            const formattedResults = [];
            for (let i = 0; i < imageTensors.length; i++) {
                const covidProb = results[i * 2 + 1];
                const normalProb = results[i * 2];
                
                formattedResults.push({
                    className: covidProb > normalProb ? 'COVID-19' : 'Normal',
                    confidence: Math.max(covidProb, normalProb),
                    probabilities: {
                        'COVID-19': covidProb,
                        'Normal': normalProb
                    },
                    confidenceLevel: this.getConfidenceLevel(Math.max(covidProb, normalProb))
                });
            }

            // Clean up
            batchTensor.dispose();
            predictions.dispose();

            return formattedResults;
        } catch (error) {
            console.error('Batch prediction error:', error);
            throw error;
        }
    }

    /**
     * Get confidence level for medical context
     */
    getConfidenceLevel(confidence) {
        if (confidence >= 0.9) return 'high';
        if (confidence >= 0.7) return 'medium';
        if (confidence >= 0.5) return 'low';
        return 'very-low';
    }

    /**
     * Generate Grad-CAM heatmap for model interpretability
     */
    async generateHeatmap(imageTensor) {
        if (!this.model) return null;

        try {
            // This is a simplified version - full Grad-CAM would be more complex
            const lastConvLayer = this.model.getLayer('conv4');
            const gradModel = tf.model({
                inputs: this.model.inputs,
                outputs: [lastConvLayer.output, this.model.output]
            });

            const [convOutput, predictions] = gradModel.predict(imageTensor.expandDims(0));
            const predictedClass = predictions.argMax(1).dataSync()[0];
            
            // Simplified heatmap generation
            const heatmap = convOutput.mean(2).squeeze();
            const normalizedHeatmap = heatmap.sub(heatmap.min()).div(heatmap.max().sub(heatmap.min()));
            
            const heatmapData = await normalizedHeatmap.data();
            
            // Clean up
            convOutput.dispose();
            predictions.dispose();
            gradModel.dispose();
            heatmap.dispose();
            normalizedHeatmap.dispose();

            return heatmapData;
        } catch (error) {
            console.warn('Heatmap generation failed:', error);
            return null;
        }
    }

    /**
     * Calculate model performance metrics
     */
    calculateMetrics(trueLabels, predictions) {
        const metrics = {
            accuracy: 0,
            precision: { 'COVID-19': 0, 'Normal': 0 },
            recall: { 'COVID-19': 0, 'Normal': 0 },
            f1Score: { 'COVID-19': 0, 'Normal': 0 },
            confusionMatrix: {
                'COVID-19': { 'COVID-19': 0, 'Normal': 0 },
                'Normal': { 'COVID-19': 0, 'Normal': 0 }
            }
        };

        // Calculate confusion matrix
        trueLabels.forEach((trueLabel, i) => {
            const predLabel = predictions[i];
            metrics.confusionMatrix[trueLabel][predLabel]++;
        });

        // Calculate metrics for each class
        this.classNames.forEach(className => {
            const tp = metrics.confusionMatrix[className][className];
            const fp = Object.values(metrics.confusionMatrix)
                .reduce((sum, row) => sum + (row[className] || 0), 0) - tp;
            const fn = Object.values(metrics.confusionMatrix[className])
                .reduce((sum, count, idx) => sum + (this.classNames[idx] !== className ? count : 0), 0);

            metrics.precision[className] = tp + fp > 0 ? tp / (tp + fp) : 0;
            metrics.recall[className] = tp + fn > 0 ? tp / (tp + fn) : 0;
            
            const precision = metrics.precision[className];
            const recall = metrics.recall[className];
            metrics.f1Score[className] = precision + recall > 0 ? 
                2 * (precision * recall) / (precision + recall) : 0;
        });

        // Overall accuracy
        const total = trueLabels.length;
        const correct = trueLabels.reduce((sum, label, i) => 
            sum + (label === predictions[i] ? 1 : 0), 0);
        metrics.accuracy = total > 0 ? correct / total : 0;

        return metrics;
    }

    /**
     * Save model to browser storage
     */
    async saveModel(name = 'covid19-classifier') {
        if (!this.model) {
            throw new Error('No model to save');
        }

        try {
            const saveResult = await this.model.save(`indexeddb://${name}`);
            console.log('Model saved successfully');
            return saveResult;
        } catch (error) {
            console.error('Error saving model:', error);
            throw error;
        }
    }

    /**
     * Load model from browser storage
     */
    async loadModel(name = 'covid19-classifier') {
        try {
            this.model = await tf.loadLayersModel(`indexeddb://${name}`);
            console.log('Model loaded successfully');
            return this.model;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
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
}

// Initialize global model instance
const covidModel = new COVID19Classifier();
