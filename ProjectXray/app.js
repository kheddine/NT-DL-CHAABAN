// app.js
/**
 * Main application controller for TensorFlow.js MNIST demo
 * Handles UI interactions, model training, and visualization
 */

class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.trainData = null;
        this.testData = null;
        this.isTraining = false;
        
        this.initializeEventListeners();
        this.updateUI();
    }

    /**
     * Initialize all UI event listeners
     */
    initializeEventListeners() {
        // File loading
        document.getElementById('load-data').addEventListener('click', () => this.onLoadData());
        document.getElementById('train').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluate').addEventListener('click', () => this.onEvaluate());
        document.getElementById('test-five').addEventListener('click', () => this.onTestFive());
        
        // Model management
        document.getElementById('save-model').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('load-model').addEventListener('click', () => this.onLoadFromFiles());
        
        // Utility functions
        document.getElementById('reset').addEventListener('click', () => this.onReset());
        document.getElementById('toggle-visor').addEventListener('click', () => this.toggleVisor());
    }

    /**
     * Load and parse MNIST data from uploaded CSV files
     */
    async onLoadData() {
        const trainFile = document.getElementById('train-csv').files[0];
        const testFile = document.getElementById('test-csv').files[0];
        
        if (!trainFile || !testFile) {
            this.showError('Please upload both train and test CSV files');
            return;
        }

        try {
            this.showStatus('Loading training data...', 'data-status');
            
            // Load training data
            this.trainData = await this.dataLoader.loadFromFile(trainFile);
            this.showStatus(`Training data loaded: ${this.trainData.xs.shape[0]} samples`, 'data-status');
            
            // Load test data
            this.showStatus('Loading test data...', 'data-status');
            this.testData = await this.dataLoader.loadFromFile(testFile);
            this.showStatus(
                `Data loaded - Train: ${this.trainData.xs.shape[0]}, Test: ${this.testData.xs.shape[0]} samples`, 
                'data-status'
            );
            
            // Enable training button
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`, 'data-status');
            console.error('Data loading error:', error);
        }
    }

    /**
     * Create and train the CNN model
     */
    async onTrain() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.updateUI();
            
            // Split training data into train/validation sets
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, 
                this.trainData.ys, 
                0.1
            );

            // Create CNN model
            this.model = this.createModel();
            this.updateModelInfo();

            // Training configuration
            const trainingConfig = {
                epochs: 10,
                batchSize: 128,
                validationData: [valXs, valYs],
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Training Metrics' }, 
                    ['loss', 'val_loss', 'acc', 'val_acc'], 
                    { callbacks: ['onEpochEnd', 'onBatchEnd'] }
                ),
                shuffle: true
            };

            this.showStatus('Starting training...', 'training-logs');
            const startTime = performance.now();

            // Execute training
            const history = await this.model.fit(trainXs, trainYs, trainingConfig);
            
            const endTime = performance.now();
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            
            // Find best validation accuracy
            const bestValAcc = Math.max(...history.history.val_acc);
            this.showStatus(
                `Training completed in ${duration}s - Best val_acc: ${(bestValAcc * 100).toFixed(2)}%`, 
                'training-logs'
            );

            // Enable evaluation and saving
            document.getElementById('evaluate').disabled = false;
            document.getElementById('test-five').disabled = false;
            document.getElementById('save-model').disabled = false;

            // Clean up temporary tensors
            tf.dispose([trainXs, trainYs, valXs, valYs]);

        } catch (error) {
            this.showError(`Training failed: ${error.message}`, 'training-logs');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            this.updateUI();
        }
    }

    /**
     * Evaluate model on test data and generate visualizations
     */
    async onEvaluate() {
        if (!this.model || !this.testData) {
            this.showError('Please train a model and load test data first');
            return;
        }

        try {
            this.showStatus('Evaluating model...', 'metrics');
            
            // Predict on test data in batches to avoid memory issues
            const testXs = this.testData.xs;
            const testYs = this.testData.ys;
            const batchSize = 1000;
            
            let predictions = [];
            let actualLabels = [];
            
            // Process test data in batches
            for (let i = 0; i < testXs.shape[0]; i += batchSize) {
                const batchXs = testXs.slice([i, 0, 0, 0], [batchSize, 28, 28, 1]);
                const batchYs = testYs.slice([i, 0], [batchSize, 10]);
                
                const batchPredictions = await this.model.predict(batchXs).array();
                const batchActual = await batchYs.argMax(1).array();
                
                predictions.push(...batchPredictions);
                actualLabels.push(...batchActual);
                
                // Clean up batch tensors
                tf.dispose([batchXs, batchYs]);
            }
            
            // Calculate overall accuracy
            const predictedLabels = predictions.map(p => p.indexOf(Math.max(...p)));
            const correct = predictedLabels.filter((pred, i) => pred === actualLabels[i]).length;
            const accuracy = (correct / actualLabels.length) * 100;
            
            this.showStatus(`Test Accuracy: ${accuracy.toFixed(2)}%`, 'metrics');
            
            // Generate confusion matrix
            this.renderConfusionMatrix(actualLabels, predictedLabels);
            
            // Generate per-class accuracy
            this.renderClassAccuracy(actualLabels, predictedLabels);
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`, 'metrics');
            console.error('Evaluation error:', error);
        }
    }

    /**
     * Test 5 random samples and display with predictions
     */
    async onTestFive() {
        if (!this.model || !this.testData) {
            this.showError('Please train a model and load test data first');
            return;
        }

        try {
            // Get random batch
            const batch = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);
            const predictions = this.model.predict(batch.xs);
            const predictedClasses = await predictions.argMax(1).array();
            const actualClasses = await batch.ys.argMax(1).array();
            
            // Display images and predictions
            this.renderTestPreview(batch.xs, actualClasses, predictedClasses);
            
            // Clean up
            tf.dispose([predictions]);
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
            console.error('Test preview error:', error);
        }
    }

    /**
     * Save model to user's downloads
     */
    async onSaveDownload() {
        if (!this.model) {
            this.showError('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-cnn');
            this.showStatus('Model saved to downloads', 'training-logs');
        } catch (error) {
            this.showError(`Save failed: ${error.message}`);
            console.error('Save error:', error);
        }
    }

    /**
     * Load model from user-selected files
     */
    async onLoadFromFiles() {
        const jsonFile = document.getElementById('upload-json').files[0];
        const weightsFile = document.getElementById('upload-weights').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...', 'model-info');
            
            // Load model using browser files
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            // Recompile the model
            this.model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.updateModelInfo();
            this.showStatus('Model loaded successfully', 'model-info');
            
            // Enable evaluation buttons if test data is available
            if (this.testData) {
                document.getElementById('evaluate').disabled = false;
                document.getElementById('test-five').disabled = false;
                document.getElementById('save-model').disabled = false;
            }
            
        } catch (error) {
            this.showError(`Model loading failed: ${error.message}`, 'model-info');
            console.error('Model loading error:', error);
        }
    }

    /**
     * Reset application state
     */
    onReset() {
        // Dispose tensors and models
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        this.isTraining = false;
        
        // Clear UI
        this.showStatus('No data loaded', 'data-status');
        this.showStatus('Training not started', 'training-logs');
        this.showStatus('No model loaded', 'model-info');
        this.showStatus('No evaluation performed', 'metrics');
        
        document.getElementById('preview-canvases').innerHTML = '';
        document.getElementById('preview-labels').innerHTML = '';
        
        // Reset file inputs
        document.getElementById('train-csv').value = '';
        document.getElementById('test-csv').value = '';
        document.getElementById('upload-json').value = '';
        document.getElementById('upload-weights').value = '';
        
        this.updateUI();
    }

    /**
     * Toggle tfjs-vis visor display
     */
    toggleVisor() {
        tfvis.visor().toggle();
    }

    /**
     * Create CNN model architecture
     * @returns {tf.Sequential} Compiled model
     */
    createModel() {
        return tf.sequential({
            layers: [
                // First convolutional block
                tf.layers.conv2d({
                    inputShape: [28, 28, 1],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                
                // Second convolutional block
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                
                // Pooling and regularization
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.dropout({ rate: 0.25 }),
                
                // Classification head
                tf.layers.flatten(),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.5 }),
                tf.layers.dense({ units: 10, activation: 'softmax' })
            ]
        });
        
        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    /**
     * Update model information display
     */
    updateModelInfo() {
        if (!this.model) {
            document.getElementById('model-info').innerHTML = 'No model loaded';
            return;
        }

        let html = '<strong>Model Architecture:</strong><br>';
        
        this.model.layers.forEach((layer, i) => {
            html += `${i + 1}. ${layer.name} (${layer.getClassName()})<br>`;
        });
        
        // Count trainable parameters
        const trainableCount = this.model.trainableWeights
            .map(w => w.size)
            .reduce((a, b) => a + b, 0);
            
        html += `<br><strong>Trainable parameters:</strong> ${trainableCount.toLocaleString()}`;
        
        document.getElementById('model-info').innerHTML = html;
    }

    /**
     * Render confusion matrix using tfjs-vis
     */
    renderConfusionMatrix(actualLabels, predictedLabels) {
        const confusionMatrix = [];
        
        // Initialize 10x10 matrix with zeros
        for (let i = 0; i < 10; i++) {
            confusionMatrix[i] = new Array(10).fill(0);
        }
        
        // Populate confusion matrix
        for (let i = 0; i < actualLabels.length; i++) {
            const actual = actualLabels[i];
            const predicted = predictedLabels[i];
            confusionMatrix[actual][predicted]++;
        }
        
        // Render with tfjs-vis
        const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
        tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        });
    }

    /**
     * Render per-class accuracy chart
     */
    renderClassAccuracy(actualLabels, predictedLabels) {
        const classAccuracy = new Array(10).fill(0);
        const classCounts = new Array(10).fill(0);
        
        // Calculate accuracy per class
        for (let i = 0; i < actualLabels.length; i++) {
            const actual = actualLabels[i];
            classCounts[actual]++;
            if (actual === predictedLabels[i]) {
                classAccuracy[actual]++;
            }
        }
        
        // Convert to percentages
        const accuracyPercentages = classAccuracy.map((correct, i) => 
            classCounts[i] > 0 ? (correct / classCounts[i]) * 100 : 0
        );
        
        // Prepare data for bar chart
        const accuracyData = accuracyPercentages.map((acc, i) => ({
            index: i,
            accuracy: acc,
            class: i.toString()
        }));
        
        // Render with tfjs-vis
        const container = { name: 'Per-Class Accuracy', tab: 'Evaluation' };
        tfvis.render.barchart(container, accuracyData, {
            xLabel: 'Class',
            yLabel: 'Accuracy (%)',
            yAxisDomain: [0, 100]
        });
    }

    /**
     * Render test preview with images and predictions
     */
    renderTestPreview(images, actualClasses, predictedClasses) {
        const canvasContainer = document.getElementById('preview-canvases');
        const labelsContainer = document.getElementById('preview-labels');
        
        canvasContainer.innerHTML = '';
        labelsContainer.innerHTML = '';
        
        // Process each sample
        for (let i = 0; i < images.shape[0]; i++) {
            const imageTensor = images.slice([i, 0, 0, 0], [1, 28, 28, 1]);
            const actual = actualClasses[i];
            const predicted = predictedClasses[i];
            const isCorrect = actual === predicted;
            
            // Create canvas for image
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            canvas.className = 'preview-canvas';
            
            // Draw image to canvas
            this.dataLoader.draw28x28ToCanvas(imageTensor, canvas, 4);
            
            // Create label display
            const labelDiv = document.createElement('div');
            labelDiv.innerHTML = `Pred: ${predicted} | Actual: ${actual}`;
            labelDiv.className = isCorrect ? 'correct' : 'wrong';
            
            previewItem.appendChild(canvas);
            previewItem.appendChild(labelDiv);
            
            canvasContainer.appendChild(previewItem);
        }
    }

    /**
     * Update UI state based on current application status
     */
    updateUI() {
        const trainBtn = document.getElementById('train');
        trainBtn.disabled = this.isTraining || !this.trainData;
        trainBtn.textContent = this.isTraining ? 'Training...' : 'Train';
        
        // Disable other buttons during training
        const otherButtons = ['evaluate', 'test-five', 'save-model', 'load-model', 'reset'];
        otherButtons.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.disabled = this.isTraining;
        });
    }

    /**
     * Show status message in specified element
     */
    showStatus(message, elementId = 'training-logs') {
        const element = document.getElementById(elementId);
        element.textContent = message;
        element.className = 'status';
    }

    /**
     * Show error message
     */
    showError(message, elementId = 'training-logs') {
        const element = document.getElementById(elementId);
        element.textContent = message;
        element.className = 'status error';
        console.error(message);
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
