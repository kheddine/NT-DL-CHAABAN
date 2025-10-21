// gru.js
class GRUModel {
    constructor() {
        this.model = null;
        this.trainingHistory = null;
        this.isTraining = false;
        this.trainingStartTime = null;
    }

    async createModel(inputShape) {
        // Ensure TensorFlow.js backend is ready
        await tf.ready();
        
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: inputShape,
                    recurrentInitializer: 'glorotNormal'
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.gru({
                    units: 32,
                    recurrentInitializer: 'glorotNormal'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 7, activation: 'linear' })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return this.model;
    }

    async trainModel(X_train, y_train, X_test, y_test, epochs = 50, callbacks = {}) {
        if (!this.model) throw new Error('Model not created');
        
        this.isTraining = true;
        this.trainingStartTime = Date.now();
        
        const history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (callbacks.onEpochEnd) {
                        callbacks.onEpochEnd(epoch, logs);
                    }
                    // Prevent memory leaks and allow UI updates
                    await tf.nextFrame();
                },
                onTrainEnd: (logs) => {
                    this.isTraining = false;
                    if (callbacks.onTrainEnd) {
                        callbacks.onTrainEnd(logs);
                    }
                }
            }
        });

        this.trainingHistory = history;
        return history;
    }

    async predict(X) {
        if (!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    evaluateModel(yTrue, yPred) {
        const mse = tf.metrics.meanSquaredError(yTrue, yPred);
        const mae = tf.metrics.meanAbsoluteError(yTrue, yPred);
        const rmse = tf.sqrt(mse);
        
        // Calculate MAPE (Mean Absolute Percentage Error)
        const absoluteError = tf.abs(tf.sub(yTrue, yPred));
        const percentageError = tf.div(absoluteError, tf.abs(yTrue).add(1e-8));
        const mape = tf.mul(tf.mean(percentageError), 100);
        
        // Calculate directional accuracy
        const trueDiffs = tf.sub(yTrue.slice([0, 1], [-1, -1]), yTrue.slice([0, 0], [-1, -1]));
        const predDiffs = tf.sub(yPred.slice([0, 1], [-1, -1]), yPred.slice([0, 0], [-1, -1]));
        
        const trueSigns = tf.sign(trueDiffs);
        const predSigns = tf.sign(predDiffs);
        const correctDirections = tf.equal(trueSigns, predSigns);
        const directionalAccuracy = tf.mul(tf.mean(correctDirections), 100);
        
        const metrics = {
            mse: mse.dataSync()[0],
            mae: mae.dataSync()[0],
            rmse: rmse.dataSync()[0],
            mape: mape.dataSync()[0],
            directionalAccuracy: directionalAccuracy.dataSync()[0]
        };
        
        // Clean up tensors
        mse.dispose();
        mae.dispose();
        rmse.dispose();
        absoluteError.dispose();
        percentageError.dispose();
        mape.dispose();
        trueDiffs.dispose();
        predDiffs.dispose();
        trueSigns.dispose();
        predSigns.dispose();
        correctDirections.dispose();
        directionalAccuracy.dispose();
        
        return metrics;
    }

    async saveModel(name = 'retail_demand_gru') {
        if (!this.model) throw new Error('No model to save');
        
        try {
            await this.model.save(`indexeddb://${name}`);
            return true;
        } catch (error) {
            console.error('Error saving model:', error);
            return false;
        }
    }

    async loadModel(name = 'retail_demand_gru') {
        try {
            await tf.ready();
            this.model = await tf.loadLayersModel(`indexeddb://${name}`);
            
            // Recompile the model
            this.model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'meanSquaredError',
                metrics: ['mae']
            });
            
            return true;
        } catch (error) {
            console.warn('No saved model found:', error.message);
            return false;
        }
    }

    getTrainingTime() {
        if (!this.trainingStartTime) return 0;
        return Date.now() - this.trainingStartTime;
    }

    getModelSummary() {
        if (!this.model) return 'No model created';
        
        let summary = 'Model Architecture:\n';
        let totalParams = 0;
        
        this.model.layers.forEach((layer, i) => {
            const layerType = layer.getClassName();
            const outputShape = JSON.stringify(layer.outputShape);
            const params = layer.countParams();
            totalParams += params;
            
            summary += `Layer ${i}: ${layerType.padEnd(15)} Output: ${outputShape.padEnd(20)} Params: ${params}\n`;
        });
        
        summary += `\nTotal Parameters: ${totalParams.toLocaleString()}`;
        return summary;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default GRUModel;
