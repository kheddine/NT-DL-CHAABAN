// gru.js
class GRUModel {
    constructor() {
        this.model = null;
    }

    async createModel(inputShape) {
        await tf.ready();
        
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 32,
                    returnSequences: false,
                    inputShape: inputShape
                }),
                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 7, activation: 'linear' })
            ]
        });

        this.model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return this.model;
    }

    async trainModel(X_train, y_train, X_test, y_test, epochs = 10, callbacks = {}) {
        return await this.model.fit(X_train, y_train, {
            epochs: epochs,
            validationData: [X_test, y_test],
            callbacks: callbacks
        });
    }

    async predict(X) {
        return this.model.predict(X);
    }

    evaluateModel(yTrue, yPred) {
        const mse = tf.metrics.meanSquaredError(yTrue, yPred);
        const mae = tf.metrics.meanAbsoluteError(yTrue, yPred);
        
        return {
            mse: mse.dataSync()[0],
            mae: mae.dataSync()[0]
        };
    }

    async saveModel() {
        try {
            await this.model.save('indexeddb://retail-model');
            return true;
        } catch (error) {
            console.warn('Save failed:', error);
            return false;
        }
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://retail-model');
            this.model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError',
                metrics: ['mae']
            });
            return true;
        } catch (error) {
            console.warn('Load failed:', error);
            return false;
        }
    }
}

export default GRUModel;
