// app.js

// Global variables
let model = null;
let trainingHistory = [];
let isTraining = false;
let currentMetrics = {
    trainAccuracy: 0,
    valAccuracy: 0,
    testAccuracy: 0,
    trainingTime: 0
};

// DOM elements
const dataStatusEl = document.getElementById('data-status');
const trainingProgressEl = document.getElementById('training-progress');
const trainingChartsEl = document.getElementById('training-charts');
const modelSummaryEl = document.getElementById('model-summary');
const testPreviewEl = document.getElementById('test-preview');
const evaluationResultsEl = document.getElementById('evaluation-results');
const metricsDisplayEl = document.getElementById('metrics-display');

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('MNIST Classifier initialized');
    updateMetricsDisplay();
    initializeModelSummary();
});

/**
 * Load and preprocess MNIST data from uploaded CSV files
 */
async function onLoadData() {
    const trainFileInput = document.getElementById('train-file');
    const testFileInput = document.getElementById('test-file');
    
    const trainFile = trainFileInput.files[0];
    const testFile = testFileInput.files[0];
    
    if (!trainFile && !testFile) {
        alert('Please upload at least one CSV file (train or test)');
        return;
    }
    
    try {
        // Show loading state
        dataStatusEl.className = 'status-panel warning';
        dataStatusEl.innerHTML = '<div class="spinner"></div>Loading and preprocessing data...';
        
        // Clear previous data
        disposeData();
        
        // Load training data if provided
        if (trainFile) {
            await loadTrainFromFiles(trainFile);
            
            // Split training data into train/validation sets
            const { trainXs, trainYs, valXs, valYs } = splitTrainVal(
                mnistData.train.xs, 
                mnistData.train.ys, 
                0.1
            );
            
            // Update data storage
            mnistData.train.xs = trainXs;
            mnistData.train.ys = trainYs;
            mnistData.validation.xs = valXs;
            mnistData.validation.ys = valYs;
        }
        
        // Load test data if provided
        if (testFile) {
            await loadTestFromFiles(testFile);
        }
        
        // Update UI with data statistics
        const stats = getDataStats();
        let statusHTML = '<strong>Data Loaded Successfully!</strong><br>';
        
        if (stats.trainSamples > 0) {
            statusHTML += `Training samples: ${stats.trainSamples.toLocaleString()}<br>`;
            statusHTML += `Validation samples: ${stats.validationSamples.toLocaleString()}<br>`;
        }
        if (stats.testSamples > 0) {
            statusHTML += `Test samples: ${stats.testSamples.toLocaleString()}<br>`;
        }
        
        statusHTML += '<small>Images normalized to [0,1] and reshaped to 28×28×1</small>';
        
        dataStatusEl.className = 'status-panel success';
        dataStatusEl.innerHTML = statusHTML;
        
        console.log('MNIST data loaded:', stats);
        
        // Enable training button if training data is available
        updateButtonStates();
        
    } catch (error) {
        console.error('Error loading data:', error);
        dataStatusEl.className = 'status-panel error';
        dataStatusEl.innerHTML = `<strong>Error loading data:</strong> ${error.message}`;
    }
}

/**
 * Create and train the CNN model
 */
async function onTrain() {
    if (!mnistData.train.xs) {
        alert('Please load training data first');
        return;
    }
    
    if (isTraining) {
        alert('Training already in progress');
        return;
    }
    
    try {
        isTraining = true;
        trainingProgressEl.style.display = 'block';
        trainingProgressEl.className = 'status-panel warning';
        trainingProgressEl.innerHTML = '<div class="spinner"></div>Initializing model training...';
        
        // Create model if it doesn't exist
        if (!model) {
            createModel();
        }
        
        // Clear previous charts
        trainingChartsEl.innerHTML = '';
        
        // Prepare training callbacks for live visualization
        const callbacks = tfvis.show.fitCallbacks(
            trainingChartsEl,
            ['loss', 'val_loss', 'acc', 'val_acc'],
            {
                callbacks: ['onEpochEnd', 'onBatchEnd'],
                height: 300,
                width: 500
            }
        );
        
        // Training configuration
        const trainingConfig = {
            epochs: 8,
            batchSize: 128,
            validationData: [mnistData.validation.xs, mnistData.validation.ys],
            shuffle: true,
            callbacks: callbacks
        };
        
        console.log('Starting model training with config:', trainingConfig);
        trainingProgressEl.innerHTML = '<div class="spinner"></div>Training started...';
        
        const startTime = Date.now();
        
        // Train the model
        const history = await model.fit(
            mnistData.train.xs, 
            mnistData.train.ys, 
            trainingConfig
        );
        
        const trainingTime = ((Date.now() - startTime) / 1000).toFixed(1);
        trainingHistory = history;
        
        // Calculate final metrics
        const finalLoss = history.history.loss[history.history.loss.length - 1].toFixed(4);
        const finalAcc = history.history.acc[history.history.acc.length - 1].toFixed(4);
        const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1].toFixed(4);
        const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1].toFixed(4);
        
        // Update global metrics
        currentMetrics.trainAccuracy = finalAcc;
        currentMetrics.valAccuracy = finalValAcc;
        currentMetrics.trainingTime = trainingTime;
        
        trainingProgressEl.className = 'status-panel success';
        trainingProgressEl.innerHTML = `
            <strong>Training Completed!</strong><br>
            Duration: ${trainingTime} seconds<br>
            Final Training Accuracy: ${(finalAcc * 100).toFixed(1)}%<br>
            Final Validation Accuracy: ${(finalValAcc * 100).toFixed(1)}%<br>
            Final Training Loss: ${finalLoss}<br>
            Final Validation Loss: ${finalValLoss}
        `;
        
        console.log(`Training completed in ${trainingTime}s. Final accuracy: ${(finalAcc * 100).toFixed(1)}%`);
        
        // Update metrics display
        updateMetricsDisplay();
        updateModelSummary();
        
    } catch (error) {
        console.error('Error during training:', error);
        trainingProgressEl.className = 'status-panel error';
        trainingProgressEl.innerHTML = `<strong>Training failed:</strong> ${error.message}`;
    } finally {
        isTraining = false;
    }
}

/**
 * Evaluate the model on test data
 */
async function onEvaluate() {
    if (!model) {
        alert('No model available. Please train or load a model first.');
        return;
    }
    
    if (!mnistData.test.xs) {
        alert('No test data available. Please load test CSV file.');
        return;
    }
    
    try {
        evaluationResultsEl.style.display = 'block';
        evaluationResultsEl.className = 'status-panel warning';
        evaluationResultsEl.innerHTML = '<div class="spinner"></div>Evaluating model on test data...';
        
        console.log('Starting model evaluation...');
        
        // Make predictions on test data
        const predictions = model.predict(mnistData.test.xs);
        const predictedClasses = predictions.argMax(-1);
        const trueClasses = mnistData.test.ys.argMax(-1);
        
        // Calculate accuracy
        const accuracy = tf.tidy(() => {
            const correctPredictions = predictedClasses.equal(trueClasses);
            return correctPredictions.mean().dataSync()[0];
        });
        
        // Calculate confusion matrix
        const confusionMatrix = await tfvis.metrics.confusionMatrix(trueClasses, predictedClasses);
        
        // Calculate per-class accuracy
        const classAccuracy = [];
        const numClasses = 10;
        
        for (let i = 0; i < numClasses; i++) {
            const classMask = trueClasses.equal(i);
            const classPredictions = tf.booleanMask(predictedClasses, classMask);
            const classCorrect = classPredictions.equal(i);
            const accuracy = classCorrect.mean().dataSync()[0];
            classAccuracy.push(accuracy);
        }
        
        // Update global metrics
        currentMetrics.testAccuracy = accuracy;
        
        // Display results
        evaluationResultsEl.className = 'status-panel success';
        evaluationResultsEl.innerHTML = `
            <strong>Evaluation Completed!</strong><br>
            Test Accuracy: ${(accuracy * 100).toFixed(2)}%<br>
            <small>Model evaluated on ${mnistData.test.xs.shape[0].toLocaleString()} test samples</small>
        `;
        
        // Visualize results in tfjs-vis visor
        const surface = { name: 'Model Evaluation', tab: 'Evaluation' };
        
        // Show confusion matrix
        tfvis.render.confusionMatrix(surface, {
            values: confusionMatrix,
            tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        });
        
        // Show per-class accuracy
        const accuracySurface = { name: 'Per-Class Accuracy', tab: 'Evaluation' };
        tfvis.render.barchart(accuracySurface, classAccuracy.map((acc, i) => ({
            index: i,
            value: acc,
            color: acc > 0.8 ? '#28a745' : acc > 0.6 ? '#ffc107' : '#dc3545'
        })), {
            xLabel: 'Digit',
            yLabel: 'Accuracy',
            yAxisDomain: [0, 1]
        });
        
        console.log(`Evaluation completed. Test accuracy: ${(accuracy * 100).toFixed(2)}%`);
        
        // Update metrics display
        updateMetricsDisplay();
        
        // Clean up tensors
        predictions.dispose();
        predictedClasses.dispose();
        trueClasses.dispose();
        
    } catch (error) {
        console.error('Error during evaluation:', error);
        evaluationResultsEl.className = 'status-panel error';
        evaluationResultsEl.innerHTML = `<strong>Evaluation failed:</strong> ${error.message}`;
    }
}

/**
 * Test 5 random samples and display predictions
 */
async function onTestFive() {
    if (!model) {
        alert('No model available. Please train or load a model first.');
        return;
    }
    
    if (!mnistData.test.xs) {
        alert('No test data available. Please load test CSV file.');
        return;
    }
    
    try {
        // Get random batch of test samples
        const { samples, labels, indices } = getRandomTestBatch(
            mnistData.test.xs, 
            mnistData.test.ys, 
            5
        );
        
        // Make predictions
        const predictions = model.predict(samples);
        const predictedClasses = predictions.argMax(-1);
        const trueClasses = labels.argMax(-1);
        
        // Get prediction probabilities
        const probabilities = predictions.dataSync();
        const predictedArray = await predictedClasses.array();
        const trueArray = await trueClasses.array();
        
        // Clear previous preview
        testPreviewEl.innerHTML = '';
        
        // Display each sample with prediction
        for (let i = 0; i < 5; i++) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            canvas.className = 'preview-canvas';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'prediction-label';
            
            // Get the image tensor for this sample
            const sampleTensor = samples.slice([i, 0, 0, 0], [1, 28, 28, 1]);
            
            // Draw image to canvas
            draw28x28ToCanvas(sampleTensor, canvas, 4);
            
            // Determine if prediction is correct
            const isCorrect = predictedArray[i] === trueArray[i];
            const confidence = probabilities[i * 10 + predictedArray[i]];
            
            labelDiv.textContent = `Pred: ${predictedArray[i]} (${(confidence * 100).toFixed(1)}%)`;
            labelDiv.className = `prediction-label ${isCorrect ? 'prediction-correct' : 'prediction-incorrect'}`;
            
            // Add true label as tooltip
            previewItem.title = `True label: ${trueArray[i]}`;
            
            previewItem.appendChild(canvas);
            previewItem.appendChild(labelDiv);
            testPreviewEl.appendChild(previewItem);
            
            // Clean up sample tensor
            sampleTensor.dispose();
        }
        
        console.log('Displayed 5 random test predictions');
        
        // Clean up tensors
        samples.dispose();
        labels.dispose();
        predictions.dispose();
        predictedClasses.dispose();
        trueClasses.dispose();
        
    } catch (error) {
        console.error('Error during random test:', error);
        testPreviewEl.innerHTML = `<div class="status-panel error">Error: ${error.message}</div>`;
    }
}

/**
 * Create the CNN model architecture for MNIST classification
 */
function createModel() {
    // Clean up previous model if exists
    if (model) {
        model.dispose();
    }
    
    model = tf.sequential({
        layers: [
            // First convolutional block
            tf.layers.conv2d({
                filters: 32,
                kernelSize: 3,
                activation: 'relu',
                padding: 'same',
                inputShape: [28, 28, 1]
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
            
            // Flatten for dense layers
            tf.layers.flatten(),
            
            // Dense layers
            tf.layers.dense({ units: 128, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.5 }),
            
            // Output layer (10 classes for digits 0-9)
            tf.layers.dense({ units: 10, activation: 'softmax' })
        ]
    });
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log('CNN model created and compiled for MNIST classification');
}

/**
 * Save the trained model to user's downloads
 */
async function onSaveModel() {
    if (!model) {
        alert('No model to save. Please train a model first.');
        return;
    }
    
    try {
        await model.save('downloads://mnist-cnn-model');
        console.log('Model saved successfully');
        alert('Model saved successfully! Check your downloads folder for model.json and weights.bin');
    } catch (error) {
        console.error('Error saving model:', error);
        alert('Error saving model: ' + error.message);
    }
}

/**
 * Load a model from user-selected files
 */
async function onLoadModel(event) {
    const files = event.target.files;
    if (!files || files.length === 0) {
        return;
    }
    
    try {
        // Separate JSON and weights files
        const jsonFile = Array.from(files).find(file => file.name.endsWith('.json'));
        const weightsFile = Array.from(files).find(file => file.name.endsWith('.bin'));
        
        if (!jsonFile || !weightsFile) {
            throw new Error('Please select both model.json and weights.bin files');
        }
        
        // Show loading message
        modelSummaryEl.textContent = 'Loading model...';
        
        // Load the model
        model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
        
        // Recompile the model
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Model loaded successfully');
        updateModelSummary();
        updateButtonStates();
        
        alert('Model loaded successfully!');
        
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Error loading model: ' + error.message);
        modelSummaryEl.textContent = 'Error loading model';
    }
    
    // Reset file input
    event.target.value = '';
}

/**
 * Reset the application - clear all data and model
 */
function onReset() {
    if (confirm('Are you sure you want to reset? This will clear all data and the current model.')) {
        // Dispose tensors and model
        if (model) {
            model.dispose();
            model = null;
        }
        disposeData();
        
        // Reset UI
        dataStatusEl.className = 'status-panel';
        dataStatusEl.textContent = 'No data loaded. Please upload MNIST CSV files and click "Load Data".';
        
        trainingProgressEl.style.display = 'none';
        evaluationResultsEl.style.display = 'none';
        trainingChartsEl.innerHTML = 'Training charts will appear here once training starts...';
        testPreviewEl.innerHTML = '';
        
        // Reset metrics
        currentMetrics = {
            trainAccuracy: 0,
            valAccuracy: 0,
            testAccuracy: 0,
            trainingTime: 0
        };
        
        updateMetricsDisplay();
        initializeModelSummary();
        updateButtonStates();
        
        console.log('Application reset');
    }
}

/**
 * Update the model summary display
 */
function updateModelSummary() {
    if (!model) {
        modelSummaryEl.textContent = 'No model loaded. Train a new model or load an existing one.';
        return;
    }
    
    try {
        let summary = 'Model Architecture:\n\n';
        let totalParams = 0;
        let trainableParams = 0;
        
        model.layers.forEach((layer, i) => {
            const layerType = layer.getClassName();
            const outputShape = JSON.stringify(layer.outputShape).slice(1, -1);
            const params = layer.countParams();
            
            totalParams += params;
            if (layer.trainable) {
                trainableParams += params;
            }
            
            summary += `${(i + 1).toString().padStart(2, ' ')}) ${layerType.padEnd(20)} ${outputShape.padEnd(25)} ${params.toString().padStart(8)} params\n`;
        });
        
        summary += `\nTotal params: ${totalParams.toLocaleString()}\n`;
        summary += `Trainable params: ${trainableParams.toLocaleString()}\n`;
        summary += `Non-trainable params: ${(totalParams - trainableParams).toLocaleString()}\n\n`;
        summary += `Input shape: [28, 28, 1]\n`;
        summary += `Output shape: [10] (digits 0-9)`;
        
        modelSummaryEl.textContent = summary;
        
    } catch (error) {
        console.error('Error generating model summary:', error);
        modelSummaryEl.textContent = 'Error generating model summary';
    }
}

/**
 * Initialize model summary with basic information
 */
function initializeModelSummary() {
    const summary = `Model Architecture (when created):

 1) conv2d              28, 28, 32              320 params
 2) conv2d              28, 28, 64            18496 params
 3) max_pooling2d       14, 14, 64                0 params
 4) dropout             14, 14, 64                0 params
 5) flatten                12544                  0 params
 6) dense                  128              1605760 params
 7) dropout                128                    0 params
 8) dense                   10                1290 params

Total params: 1,625,866
Trainable params: 1,625,866
Non-trainable params: 0

Input shape: [28, 28, 1]
Output shape: [10] (digits 0-9)`;
    
    modelSummaryEl.textContent = summary;
}

/**
 * Update the metrics display with current performance data
 */
function updateMetricsDisplay() {
    let metricsHTML = '';
    
    if (currentMetrics.trainAccuracy > 0) {
        metricsHTML += `
            <div class="metric-card">
                <div class="metric-label">Training Accuracy</div>
                <div class="metric-value">${(currentMetrics.trainAccuracy * 100).toFixed(1)}%</div>
            </div>
        `;
    }
    
    if (currentMetrics.valAccuracy > 0) {
        metricsHTML += `
            <div class="metric-card">
                <div class="metric-label">Validation Accuracy</div>
                <div class="metric-value">${(currentMetrics.valAccuracy * 100).toFixed(1)}%</div>
            </div>
        `;
    }
    
    if (currentMetrics.testAccuracy > 0) {
        metricsHTML += `
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">${(currentMetrics.testAccuracy * 100).toFixed(1)}%</div>
            </div>
        `;
    }
    
    if (currentMetrics.trainingTime > 0) {
        metricsHTML += `
            <div class="metric-card">
                <div class="metric-label">Training Time</div>
                <div class="metric-value">${currentMetrics.trainingTime}s</div>
            </div>
        `;
    }
    
    metricsDisplayEl.innerHTML = metricsHTML || `
        <div class="metric-card">
            <div class="metric-label">No Metrics Yet</div>
            <div class="metric-value">-</div>
        </div>
    `;
}

/**
 * Update button states based on available data and model
 */
function updateButtonStates() {
    const hasTrainData = mnistData.train.xs !== null;
    const hasTestData = mnistData.test.xs !== null;
    const hasModel = model !== null;
    
    // You can implement button enabling/disabling logic here if needed
    console.log('UI state - Train data:', hasTrainData, 'Test data:', hasTestData, 'Model:', hasModel);
}

// Handle page unload to clean up memory
window.addEventListener('beforeunload', () => {
    if (model) {
        model.dispose();
    }
    disposeData();
});
