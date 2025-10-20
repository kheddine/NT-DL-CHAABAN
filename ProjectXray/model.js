class PneumoniaModel {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.inputSize = [150, 150];
        this.classNames = ['Normal', 'Pneumonia'];
    }

    async initialize() {
        try {
            await tf.setBackend('webgl');
            console.log('Using WebGL backend');
        } catch (error) {
            await tf.setBackend('cpu');
            console.log('Using CPU backend');
        }
    }

    createModel() {
        const model = tf.sequential();
        
        // Input: 150x150x3 RGB images
        model.add(tf.layers.conv2d({
            inputShape: [this.inputSize[0], this.inputSize[1], 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        
        // Classification head
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ 
            units: 128, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ 
            units: 64, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.dropout({ rate: 0.3 }));
        model.add(tf.layers.dense({ 
            units: 2, 
            activation: 'softmax' 
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log('CNN Model created');
        return model;
    }

    async train(trainingData, epochs, batchSize, onProgress) {
        if (!this.model) {
            this.model = this.createModel();
        }

        this.isTraining = true;
        
        try {
            const history = await this.model.fit(trainingData.xs, trainingData.ys, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        if (onProgress) {
                            onProgress({
                                epoch: epoch + 1,
                                totalEpochs: epochs,
                                accuracy: logs.acc,
                                loss: logs.loss,
                                valAccuracy: logs.val_acc,
                                valLoss: logs.val_loss
                            });
                        }
                        await tf.nextFrame();
                    }
                }
            });
            
            await this.saveModel();
            return history;
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        } finally {
            this.isTraining = false;
        }
    }

    async predict(imageTensor) {
        if (!this.model) {
            throw new Error('No trained model available. Please train a model first.');
        }

        const batched = imageTensor.expandDims(0);
        const prediction = this.model.predict(batched);
        const probabilities = await prediction.data();
        
        batched.dispose();
        prediction.dispose();
        
        return {
            normal: probabilities[0],
            pneumonia: probabilities[1],
            predictedClass: probabilities[1] > probabilities[0] ? 'Pneumonia' : 'Normal',
            confidence: Math.max(probabilities[0], probabilities[1])
        };
    }

    async predictBatch(imageTensors) {
        const predictions = [];
        for (let i = 0; i < imageTensors.length; i++) {
            try {
                const prediction = await this.predict(imageTensors[i]);
                predictions.push(prediction);
                
                // Prevent memory issues
                if (i % 5 === 0) {
                    await tf.nextFrame();
                }
            } catch (error) {
                console.error(`Prediction failed for image ${i}:`, error);
                predictions.push(null);
            }
        }
        return predictions;
    }

    async saveModel() {
        if (this.model) {
            try {
                await this.model.save('indexeddb://pneumonia-model');
                console.log('Model saved successfully');
                return true;
            } catch (error) {
                console.error('Failed to save model:', error);
                return false;
            }
        }
        return false;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://pneumonia-model');
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.log('No saved model found');
            return false;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }

    getMemoryStats() {
        return tf.memory();
    }
}
