// gru.js
class GRUModel {
    constructor() {
        this.model = null;
        this.trainingHistory = null;
        this.isTraining = false;
    }

    createModel(inputShape) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: inputShape
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
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

    async trainModel(X_train, y_train, X_test, y_test, epochs = 100, callbacks = {}) {
        if (!this.model) throw new Error('Model not created');
        
        this.isTraining = true;
        
        const history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (callbacks.onEpochEnd) {
                        callbacks.onEpochEnd(epoch, logs);
                    }
                    // Prevent memory leaks
                    await tf.nextFrame();
                },
                onTrainEnd: () => {
                    this.isTraining = false;
                    if (callbacks.onTrainEnd) {
                        callbacks.onTrainEnd();
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
        
        return {
            mse: mse.dataSync()[0],
            mae: mae.dataSync()[0],
            rmse: rmse.dataSync()[0]
        };
    }

    async saveModel(name = 'demand_forecast_model') {
        if (!this.model) throw new Error('No model to save');
        
        const saveResult = await this.model.save(`indexeddb://${name}`);
        return saveResult;
    }

    async loadModel(name = 'demand_forecast_model') {
        try {
            this.model = await tf.loadLayersModel(`indexeddb://${name}`);
            return true;
        } catch (error) {
            console.warn('No saved model found:', error);
            return false;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    getModelSummary() {
        if (!this.model) return 'No model created';
        
        let summary = '';
        this.model.layers.forEach((layer, i) => {
            summary += `Layer ${i}: ${layer.getClassName()} - ${layer.outputShape}\n`;
        });
        return summary;
    }
}

export default GRUModel;
