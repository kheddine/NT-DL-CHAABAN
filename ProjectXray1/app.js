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

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('COVID-19 X-ray Classifier initialized');
    updateModelSummary();
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
        dataStatusEl.innerHTML = 'Loading and preprocessing images...';
        
        // Prepare training data
        const { xs, ys } = await prepareTrainingData(pneumoniaFiles, normalFiles);
        
        // Store in global variable
        trainingData.xs = xs;
        trainingData.ys = ys;
        
        // Update UI with data statistics
        const stats = getTrainingDataStats();
        dataStatusEl.innerHTML = `
            <strong>Data Loaded Successfully!</strong><br>
            Pneumonia images: ${stats.pneumoniaCount}<br>
            Normal images: ${stats.normalCount}<br>
            Total samples: ${stats.totalSamples}<br>
            Input shape: ${stats.inputShape.join(' × ')}<br>
            <small>Images resized to 128×128 pixels and normalized</small>
        `;
        
        console.log('Training data prepared:', stats);
        
    } catch (error) {
        console.error('Error loading training data:', error);
        dataStatusEl.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
    }
}

/**
 * Create and train the CNN model
 */
async function onTrain() {
    if (!trainingData.xs || !trainingData.ys) {
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
        trainingProgressEl.innerHTML = 'Initializing model training...';
        
        // Create model if it doesn't exist
        if (!model) {
            createModel();
        }
        
        // Prepare training callbacks for live visualization
        const callbacks = tfvis.show.fitCallbacks(
            trainingChartsEl,
            ['loss', 'val_loss', 'acc', 'val_acc'],
            {
                callbacks: ['onBatchEnd', 'onEpochEnd'],
                height: 300,
                width: 400
            }
        );
        
        // Training configuration
        const trainingConfig = {
            epochs: 8,
            batchSize: Math.min(16, Math.floor(trainingData.xs.shape[0] / 4)),
            validationSplit: 0.2,
            shuffle: true,
            callbacks: callbacks
        };
        
        console.log('Starting model training with config:', trainingConfig);
        trainingProgressEl.innerHTML = 'Training started...';
        
        const startTime = Date.now();
        
        // Train the model
        const history = await model.fit(trainingData.xs, trainingData.ys, trainingConfig);
        
        const trainingTime = ((Date.now() - startTime) / 1000).toFixed(1);
        trainingHistory = history;
        
        // Display final metrics
        const finalLoss = history.history.loss[history.history.loss.length - 1].toFixed(4);
        const finalAcc = history.history.acc[history.history.acc.length - 1].toFixed(4);
        const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1].toFixed(4);
        const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1].toFixed(4);
        
        trainingProgressEl.innerHTML = `
            <strong>Training Completed!</strong><br>
            Duration: ${trainingTime} seconds<br>
            Final Accuracy: ${(finalAcc * 100).toFixed(1)}%<br>
            Final Validation Accuracy: ${(finalValAcc * 100).toFixed(1)}%<br>
            Final Loss: ${finalLoss}<br>
            Final Validation Loss: ${finalValLoss}
        `;
        
        console.log(`Training completed in ${trainingTime}s. Final accuracy: ${(finalAcc * 100).toFixed(1)}%`);
        
        // Update model summary
        updateModelSummary();
        
    } catch (error) {
        console.error('Error during training:', error);
        trainingProgressEl.innerHTML = `<span style="color: red;">Training failed: ${error.message}</span>`;
    } finally {
        isTraining = false;
    }
}

/**
 * Create the CNN model architecture
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
                inputShape: [128, 128, 3]
            }),
            tf.layers.maxPooling2d({ poolSize: 2 }),
            
            // Second convolutional block
            tf.layers.conv2d({
                filters: 64,
                kernelSize: 3,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({ poolSize: 2 }),
            
            // Third convolutional block
            tf.layers.conv2d({
                filters: 64,
                kernelSize: 3,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({ poolSize: 2 }),
            
            // Classification head
            tf.layers.flatten(),
            tf.layers.dense({ units: 128, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.5 }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
    });
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log('CNN model created and compiled');
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
        await model.save('downloads://xray-cnn-model');
        console.log('Model saved successfully');
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
    }
    
    // Reset file input
    event.target.value = '';
}

/**
 * Make prediction on a test image
 */
async function onPredict() {
    const testFileInput = document.getElementById('test-image');
    const testFile = testFileInput.files[0];
    
    if (!testFile) {
        alert('Please select a test image first');
        return;
    }
    
    if (!model) {
        alert('No model available. Please train or load a model first.');
        return;
    }
    
    try {
        predictionResultEl.innerHTML = 'Processing image...';
        
        // Display test image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            testImagePreviewEl.src = e.target.result;
            testImagePreviewEl.style.display = 'block';
        };
        reader.readAsDataURL(testFile);
        
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
        predictionResultEl.className = isPneumonia ? 'pneumonia' : 'normal';
        predictionResultEl.innerHTML = `
            <strong>Prediction: ${className}</strong><br>
            Confidence: ${confidencePercent}%<br>
            <small>Raw probability: ${probability.toFixed(4)}</small>
        `;
        
        console.log(`Prediction: ${className} (${confidencePercent}% confidence)`);
        
        // Clean up tensors
        testTensor.dispose();
        prediction.dispose();
        
    } catch (error) {
        console.error('Error during prediction:', error);
        predictionResultEl.innerHTML = `<span style="color: red;">Prediction failed: ${error.message}</span>`;
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
        let summary = '';
        
        // Count total parameters
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
            
            summary += `${(i + 1).toString().padStart(2, ' ')}) ${layerType.padEnd(20)} ${outputShape.padEnd(20)} ${params.toString().padStart(8)} params\n`;
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

// Handle page unload to clean up memory
window.addEventListener('beforeunload', () => {
    if (model) {
        model.dispose();
    }
    if (trainingData.xs) {
        trainingData.xs.dispose();
    }
    if (trainingData.ys) {
        trainingData.ys.dispose();
    }
});
