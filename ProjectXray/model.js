/**
 * Pneumonia Chest X-ray Classification Model
 * Convolutional Neural Network for medical image analysis
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
    }

    /**
     * Create CNN model for chest X-ray pneumonia classification
     * Architecture optimized for medical image analysis
     */
    createModel(useTransferLearning = false) {
        if (useTransferLearning) {
            return this.createTransferLearningModel();
        }
        
        const model = tf.sequential();
        
        // Input layer: 224x224x3 (RGB image)
        model.add(tf.layers.conv2d({
            inputShape: [224, 224, 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv1'
        }));
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool1'
        }));
        
        // Second convolutional block
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv2'
        }));
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool2'
        }));
        
        // Third convolutional block
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv3'
        }));
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool3'
        }));
        
        // Fourth convolutional block for deeper feature extraction
        model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv4'
        }));
        model.add(tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
            name: 'pool4'
        }));
        
        // Flatten and dense layers
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'dense1'
        }));
        model.add(tf.layers.dropout({
            rate: 0.5,  // Dropout for regularization
            name: 'dropout1'
        }));
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'dense2'
        }));
        model.add(tf.layers.dropout({
            rate: 0.3,
            name: 'dropout2'
        }));
        
        // Output layer: 2 classes (Normal, Pneumonia)
        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax',
            name: 'output'
        }));
        
        // Compile model with medical imaging considerations
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('CNN Model created successfully for pneumonia classification');
        return model;
    }

    /**
     * Create model using MobileNetV2 for transfer learning
     */
    async createTransferLearningModel() {
        // Load MobileNetV2 base model
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json'
        );
        
        // Remove the top classification layer
        const baseModel = tf.model({
            inputs: mobilenet.inputs,
            outputs: mobilenet.layers[mobilenet.layers.length - 3].output
        });
        
        // Freeze base model layers for transfer learning
        baseModel.trainable = false;
        
        // Add custom classification head for pneumonia detection
        const model = tf.sequential();
        model.add(baseModel);
        model.add(tf.layers.globalAveragePooling2d());
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax'
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.0001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Transfer learning model created successfully for pneumonia classification');
        return model;
    }

    /**
     * Train the model with medical imaging data
     * Includes validation split and callbacks for visualization
     */
    async trainModel(trainingData, validationData, config) {
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
            // Ensure tensor has correct shape [1, 224, 224, 3]
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
        
        for (const tensor of imageTensors) {
            try {
                const prediction = await this.predict(tensor);
                predictions.push(prediction);
            } catch (error) {
                console.error('Batch prediction error for tensor:', error);
                predictions.push(null);
            }
        }
        
        return predictions;
    }

    /**
     * Generate Grad-CAM heatmap for model interpretability
     * Shows which regions the model focuses on for prediction
     */
    async generateHeatmap(imageTensor, className = 'Pneumonia') {
        if (!this.model) return null;

        try {
            // Get the last convolutional layer
            const convLayer = this.model.getLayer('conv4');
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

            // Get the class index (1 for Pneumonia, 0 for Normal)
            const classIdx = className === 'Pneumonia' ? 1 : 0;
            const output = predictions.gather([classIdx], 1);
            
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
                [224, 224]
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
            const trueLabel = trueLabels[idx];
            const predLabel = pred.predictedClass;
            
            if (trueLabel === 'Pneumonia' && predLabel === 'Pneumonia') tp++;
            else if (trueLabel === 'Normal' && predLabel === 'Normal') tn++;
            else if (trueLabel === 'Normal' && predLabel === 'Pneumonia') fp++;
            else if (trueLabel === 'Pneumonia' && predLabel === 'Normal') fn++;
        });
        
        const accuracy = (tp + tn) / (tp + tn + fp + fn);
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
     * Dispose model to free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PneumoniaClassifier;
}
