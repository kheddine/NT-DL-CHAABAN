class GRUModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = [];
    }

    createModel(inputShape, learningRate = 0.001) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: inputShape
                }),
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
                }),
                tf.layers.dense({
                    units: 16,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 7,
                    activation: 'linear'
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return this.model;
    }

    async trainModel(trainSequences, trainTargets, epochs = 50, batchSize = 32, validationSplit = 0.1) {
        if (!this.model) {
            throw new Error('Model not created. Call createModel() first.');
        }

        // Convert to tensors
        const xs = tf.tensor3d(trainSequences);
        const ys = tf.tensor2d(trainTargets);

        const history = await this.model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: validationSplit,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.trainingHistory.push(logs);
                    // Dispatch custom event for UI updates
                    window.dispatchEvent(new CustomEvent('trainingProgress', {
                        detail: {
                            epoch: epoch + 1,
                            totalEpochs: epochs,
                            loss: logs.loss,
                            valLoss: logs.val_loss
                        }
                    }));
                }
            }
        });

        // Cleanup tensors
        xs.dispose();
        ys.dispose();

        this.isTrained = true;
        return history;
    }

    async predict(sequences) {
        if (!this.isTrained) {
            throw new Error('Model not trained. Call trainModel() first.');
        }

        const xs = tf.tensor3d(sequences);
        const predictions = this.model.predict(xs);
        const results = await predictions.array();
        
        xs.dispose();
        predictions.dispose();
        
        return results;
    }

    async evaluate(testSequences, testTargets) {
        if (!this.isTrained) {
            throw new Error('Model not trained. Call trainModel() first.');
        }

        const xs = tf.tensor3d(testSequences);
        const ys = tf.tensor2d(testTargets);

        const evaluation = this.model.evaluate(xs, ys);
        const loss = await evaluation[0].data();
        const mae = await evaluation[1].data();

        xs.dispose();
        ys.dispose();
        evaluation[0].dispose();
        evaluation[1].dispose();

        return {
            loss: loss[0],
            mae: mae[0]
        };
    }

    async saveModel() {
        if (!this.isTrained) return null;
        
        const savedModel = await this.model.save('indexeddb://retail-demand-gru');
        return savedModel;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://retail-demand-gru');
            this.isTrained = true;
            return true;
        } catch (error) {
            console.log('No saved model found:', error);
            return false;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    getModelSummary() {
        if (!this.model) return 'Model not created';
        let summary = [];
        this.model.layers.forEach((layer, i) => {
            summary.push(`Layer ${i}: ${layer.name} - ${layer.outputShape}`);
        });
        return summary.join('\n');
    }
}

export default GRUModel;
