class GRUModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = [];
        this.isInitialized = false;
    }

    async initialize() {
        if (this.isInitialized) return;
        
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        console.log('TensorFlow.js backend:', tf.getBackend());
        this.isInitialized = true;
    }

    createModel(inputShape, learningRate = 0.001) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 32, // Reduced for browser performance
                    returnSequences: true,
                    inputShape: inputShape
                }),
                tf.layers.gru({
                    units: 16, // Reduced for browser performance
                    returnSequences: false
                }),
                tf.layers.dense({
                    units: 8, // Reduced for browser performance
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

        console.log('Model created with input shape:', inputShape);
        return this.model;
    }

    async trainModel(trainSequences, trainTargets, epochs = 20, batchSize = 32, validationSplit = 0.1) {
        if (!this.model) {
            throw new Error('Model not created. Call createModel() first.');
        }

        await this.initialize();

        console.log('Starting training...');
        
        // Convert to tensors
        const xs = tf.tensor3d(trainSequences);
        const ys = tf.tensor2d(trainTargets);

        try {
            const history = await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: Math.min(batchSize, trainSequences.length),
                validationSplit: validationSplit,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.trainingHistory.push(logs);
                        // Dispatch custom event for UI updates
                        if (window.dispatchEvent) {
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
                }
            });

            this.isTrained = true;
            return history;
        } finally {
            // Cleanup tensors
            xs.dispose();
            ys.dispose();
        }
    }

    async predict(sequences) {
        if (!this.isTrained) {
            throw new Error('Model not trained. Call trainModel() first.');
        }

        await this.initialize();

        const xs = tf.tensor3d(sequences);
        try {
            const predictions = this.model.predict(xs);
            const results = await predictions.array();
            return results;
        } finally {
            xs.dispose();
        }
    }

    async evaluate(testSequences, testTargets) {
        if (!this.isTrained) {
            throw new Error('Model not trained. Call trainModel() first.');
        }

        await this.initialize();

        const xs = tf.tensor3d(testSequences);
        const ys = tf.tensor2d(testTargets);

        try {
            const evaluation = this.model.evaluate(xs, ys);
            const loss = await evaluation[0].data();
            const mae = await evaluation[1].data();

            return {
                loss: loss[0],
                mae: mae[0]
            };
        } finally {
            xs.dispose();
            ys.dispose();
        }
    }

    async saveModel() {
        if (!this.isTrained) return null;
        
        try {
            const savedModel = await this.model.save('indexeddb://retail-demand-gru');
            console.log('Model saved to IndexedDB');
            return savedModel;
        } catch (error) {
            console.warn('Could not save model:', error);
            return null;
        }
    }

    async loadModel() {
        await this.initialize();
        
        try {
            this.model = await tf.loadLayersModel('indexeddb://retail-demand-gru');
            this.isTrained = true;
            console.log('Model loaded from IndexedDB');
            return true;
        } catch (error) {
            console.log('No saved model found:', error.message);
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
            summary.push(`Layer ${i}: ${layer.name} - ${JSON.stringify(layer.outputShape)}`);
        });
        return summary.join('\n');
    }
}
