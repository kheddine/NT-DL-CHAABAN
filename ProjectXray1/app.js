// app.js

// Global variables
let model = null;
let trainingHistory = [];
let isTraining = false;

// DOM elements
const dataStatusEl = document.getElementById('data-status');
const trainingProgressEl = document.getElementById('training-progress');
const trainingChartsEl = document.getElementById('training-charts');
const modelSummaryEl = document.getElementById('model-summary');
const testImagePreviewEl = document.getElementById('test-image-preview');
const predictionResultEl = document.getElementById('prediction-result');
const samplePredictionsEl = document.getElementById('sample-predictions');

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('COVID-19 X-ray Classifier initialized');
    initializeModelSummary();
});

/**
 * Load and preprocess training data from uploaded files
 */
async function onLoadData() {
    const pneumoniaFiles = document.getElementById('pneumonia-files').files;
    const normalFiles = document.getElementById('normal-files').files;

    if (pneumoniaFiles.length === 0 && normalFiles.length === 0) {
        alert('Please upload at least one image for pneumonia or normal class');
        return;
    }

    try {
        // Show loading state
        dataStatusEl.className = 'status-panel';
        dataStatusEl.innerHTML = 'Loading and preprocessing X-ray images...';
        
        // Clear previous data
        disposeData();
        
        // Show immediate feedback
        await tf.nextFrame();
        
        // Prepare training data
        const { xs, ys } = await prepareTrainingData(pneumoniaFiles, normalFiles);
        
        // Store in global variable
        trainingData.xs = xs;
        trainingData.ys = ys;
        trainingData.pneumoniaFiles = pneumoniaFiles;
        trainingData.normalFiles = normalFiles;
        
        // Update UI with data statistics
        const stats = getTrainingDataStats();
        let statusHTML = '<strong>Data Loaded Successfully!</strong><br>';
        
        statusHTML += `Pneumonia X-rays: ${stats.pneumoniaCount}<br>`;
        statusHTML += `Normal X-rays: ${stats.normalCount}<br>`;
        statusHTML += `Total samples: ${stats.totalSamples}<br>`;
        statusHTML += `<small>Images resized to 128×128 pixels and normalized to [0,1]</small>`;
        
        dataStatusEl.innerHTML = statusHTML;
        
        console.log('Training data prepared:', stats);
        
        // Show sample images
        showSampleTrainingImages();
        
    } catch (error) {
        console.error('Error loading training data:', error);
        dataStatusEl.className = 'status-panel';
        dataStatusEl.innerHTML = `<strong>Error loading data:</strong> ${error.message}`;
    }
}

/**
 * Show sample training images in the UI
 */
function showSampleTrainingImages() {
    samplePredictionsEl.innerHTML = '<h4>Sample Training Images:</h4>';
    
    tf.tidy(() => {
        if (!trainingData.xs || trainingData.xs.shape[0] === 0) return;
        
        // Take up to 3 samples for display
        const sampleSize = Math.min(3, trainingData.xs.shape[0]);
        
        for (let i = 0; i < sampleSize; i++) {
            const sample = trainingData.xs.slice([i, 0, 0, 0], [1, 128, 128, 3]);
            const label = trainingData.ys.slice([i, 0], [1, 1]).dataSync()[0];
            const className = label === 1 ? 'Pneumonia' : 'Normal';
            
            createImagePreview(sample, className, label === 1);
        }
    });
}

/**
 * Create image preview element
 */
function createImagePreview(tensor, label, isPneumonia) {
    const previewItem = document.createElement('div');
    previewItem.className = 'preview-item';
    
    const canvas = document.createElement('canvas');
    canvas.className = 'preview-canvas';
    canvas.width = 64;
    canvas.height = 64;
    
    const labelDiv = document.createElement('div');
    labelDiv.className = `prediction-label ${isPneumonia ? 'pneumonia' : 'normal'}`;
    labelDiv.textContent = label;
    
    // Draw image to canvas with smaller size
    const resizedTensor = tf.tidy(() => 
        tensor.squeeze().resizeNearestNeighbor([64, 64])
    );
    tf.browser.toPixels(resizedTensor, canvas);
    resizedTensor.dispose();
    
    previewItem.appendChild(canvas);
    previewItem.appendChild(labelDiv);
    samplePredictionsEl.appendChild(previewItem);
}

/**
 * Create and train the CNN model
 */
async function onTrain() {
    if (!trainingData.xs) {
        alert('Please load training data first');
        return;
    }
    
    if (trainingData.xs.shape[0] < 10) {
        alert('Need at least 10 training samples to train effectively');
        return;
    }
    
    if (isTraining) {
        alert('Training already in progress');
        return;
    }
    
    try {
        isTraining = true;
        trainingProgressEl.style.display = 'block';
        trainingProgressEl.innerHTML = 'Initializing model training...';
        
        // Create model if it doesn't exist
        if (!model) {
            createModel();
        }
        
        // Clear previous charts
        trainingChartsEl.innerHTML = 'Training charts will appear here...';
        
        // Split data for validation
        const { trainXs, trainYs, valXs, valYs } = splitTrainVal(trainingData.xs, trainingData.ys, 0.2);
        
        // Prepare training callbacks for live visualization
        const callbacks = tfvis.show.fitCallbacks(
            trainingChartsEl,
            ['loss', 'val_loss', 'acc', 'val_acc'],
            {
                callbacks: ['onEpochEnd'],
                height: 300,
                width: 400
            }
        );
        
        // Training configuration
        const trainingConfig = {
            epochs: 5,
            batchSize: 8,
            validationData: [valXs, valYs],
            shuffle: true,
            callbacks: callbacks
        };
        
        console.log('Starting model training with config:', trainingConfig);
        trainingProgressEl.innerHTML = 'Training started...';
        
        const startTime = Date.now();
        
        // Train the model
        const history = await model.fit(trainXs, trainYs, trainingConfig);
        
        const trainingTime = ((Date.now() - startTime) / 1000).toFixed(1);
        trainingHistory = history;
        
        // Calculate final metrics
        const finalLoss = history.history.loss[history.history.loss.length - 1].toFixed(4);
        const finalAcc = history.history.acc[history.history.acc.length - 1].toFixed(4);
        const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1].toFixed(4);
        const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1].toFixed(4);
        
        trainingProgressEl.innerHTML = `
            <strong>Training Completed!</strong><br>
            Duration: ${trainingTime} seconds<br>
            Training Accuracy: ${(finalAcc * 100).toFixed(1)}%<br>
            Validation Accuracy: ${(finalValAcc * 100).toFixed(1)}%<br>
            Training Loss: ${finalLoss}<br>
            Validation Loss: ${finalValLoss}
        `;
        
        console.log(`Training completed in ${trainingTime}s. Final accuracy: ${(finalAcc * 100).toFixed(1)}%`);
        
        // Update model summary
        updateModelSummary();
        
        // Clean up tensors
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();
        
    } catch (error) {
        console.error('Error during training:', error);
        trainingProgressEl.innerHTML = `<strong>Training failed:</strong> ${error.message}`;
    } finally {
        isTraining = false;
    }
}

/**
 * Create the CNN model architecture for X-ray classification
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
                filters: 16,
                kernelSize: 3,
                activation: 'relu',
                inputShape: [128, 128, 3]
            }),
            tf.layers.maxPooling2d({ poolSize: 2 }),
            
            // Second convolutional block
            tf.layers.conv2d({
                filters: 32,
                kernelSize: 3,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({ poolSize: 2 }),
            
            // Classification head
            tf.layers.flatten(),
            tf.layers.dense({ units: 64, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.3 }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
    });
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log('CNN model created and compiled for X-ray classification');
}

/**
 * Make prediction on a test image
 */
async function onPredict() {
    const testFileInput = document.getElementById('test-image');
    const testFile = testFileInput.files[0];
    
    if (!testFile) {
        alert('Please select a test X-ray image first');
        return;
    }
    
    if (!model) {
        alert('No model available. Please train or load a model first.');
        return;
    }
    
    try {
        predictionResultEl.innerHTML = 'Processing X-ray image...';
        
        // Load and preprocess test image
        const testTensor = await loadTestImage(testFile);
        
        // Make prediction
        const prediction = model.predict(testTensor);
        const probability = await prediction.dataSync()[0];
        
        // Determine class and confidence
        const isPneumonia = probability >= 0.5;
        const className = isPneumonia ? 'Pneumonia' : 'Normal';
        const confidence = isPneumonia ? probability : (1 - probability);
        const confidencePercent = (confidence * 100).toFixed(1);
        
        // Display result
        predictionResultEl.className = `status-panel ${isPneumonia ? 'pneumonia' : 'normal'}`;
        predictionResultEl.innerHTML = `
            <strong>Prediction: ${className}</strong><br>
            Confidence: ${confidencePercent}%<br>
            <small>Probability score: ${probability.toFixed(4)}</small>
        `;
        
        console.log(`Prediction: ${className} (${confidencePercent}% confidence)`);
        
        // Clean up tensors
        testTensor.dispose();
        prediction.dispose();
        
    } catch (error) {
        console.error('Error during prediction:', error);
        predictionResultEl.innerHTML = `<strong>Prediction failed:</strong> ${error.message}`;
    }
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
        await model.save('downloads://covid-xray-cnn-model');
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
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Model loaded successfully');
        updateModelSummary();
        
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
            
            summary += `${(i + 1).toString().padStart(2, ' ')}) ${layerType.padEnd(15)} ${outputShape.padEnd(20)} ${params.toString().padStart(8)} params\n`;
        });
        
        summary += `\nTotal params: ${totalParams.toLocaleString()}\n`;
        summary += `Trainable params: ${trainableParams.toLocaleString()}\n`;
        summary += `Non-trainable params: ${(totalParams - trainableParams).toLocaleString()}`;
        
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

 1) conv2d          126, 126, 16      448 params
 2) max_pooling2d   63, 63, 16          0 params
 3) conv2d          61, 61, 32       4640 params
 4) max_pooling2d   30, 30, 32          0 params
 5) flatten         28800                0 params
 6) dense           64             1843264 params
 7) dropout         64                  0 params
 8) dense           1                   65 params

Total params: 1,848,417
Trainable params: 1,848,417
Non-trainable params: 0`;
    
    modelSummaryEl.textContent = summary;
}

// Handle page unload to clean up memory
window.addEventListener('beforeunload', () => {
    if (model) {
        model.dispose();
    }
    disposeData();
});
