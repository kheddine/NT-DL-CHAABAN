class COVID19Model {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.classNames = ['Normal', 'COVID-19'];
    }

    createModel() {
        const model = tf.sequential();

        // Input layer
        model.add(tf.layers.conv2d({
            inputShape: [224, 224, 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        // Second conv block
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        // Third conv block
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        // Flatten and dense layers
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        this.model = model;
        console.log('Model created successfully');
        return model;
    }

    async trainModel(trainingData, validationData, config) {
        if (this.isTraining) {
            throw new Error('Model is already training');
        }

        this.isTraining = true;

        const history = await this.model.fit(trainingData.xs, trainingData.ys, {
            epochs: config.epochs,
            batchSize: config.batchSize,
            validationData: [validationData.xs, validationData.ys],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (config.onEpochEnd) {
                        config.onEpochEnd(epoch, logs);
                    }
                },
                onTrainEnd: () => {
                    this.isTraining = false;
                    if (config.onTrainEnd) {
                        config.onTrainEnd();
                    }
                }
            }
        });

        return history;
    }

    async predict(imageTensor) {
        if (!this.model) {
            throw new Error('Model not trained yet');
        }

        const prediction = this.model.predict(imageTensor.expandDims(0));
        const probabilities = await prediction.data();
        
        prediction.dispose();

        return {
            className: probabilities[1] > probabilities[0] ? 'COVID-19' : 'Normal',
            confidence: Math.max(probabilities[0], probabilities[1]),
            probabilities: {
                'Normal': probabilities[0],
                'COVID-19': probabilities[1]
            }
        };
    }

    async predictBatch(imageTensors) {
        if (!this.model) {
            throw new Error('Model not trained yet');
        }

        const batchTensor = tf.stack(imageTensors);
        const predictions = this.model.predict(batchTensor);
        const results = await predictions.data();
        
        const formattedResults = [];
        for (let i = 0; i < imageTensors.length; i++) {
            const normalProb = results[i * 2];
            const covidProb = results[i * 2 + 1];
            
            formattedResults.push({
                className: covidProb > normalProb ? 'COVID-19' : 'Normal',
                confidence: Math.max(normalProb, covidProb),
                probabilities: {
                    'Normal': normalProb,
                    'COVID-19': covidProb
                }
            });
        }

        batchTensor.dispose();
        predictions.dispose();

        return formattedResults;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
