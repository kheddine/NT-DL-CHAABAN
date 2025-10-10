// Titanic Survival Classifier - TensorFlow.js
// Global variables
let rawTrainData = [];
let rawTestData = [];
let processedTrainData = null;
let processedTestData = null;
let model = null;
let trainingHistory = [];
let validationPredictions = null;
let validationLabels = null;
let testPredictions = null;
let currentThreshold = 0.5;
let isTraining = false;

// Data loading and inspection
async function loadData() {
    const trainFile = document.getElementById('trainFile').files[0];
    
    if (!trainFile) {
        alert('Please select a training CSV file (train.csv)');
        return;
    }

    try {
        showStatus('dataStatus', 'Loading training data...', 'info');
        
        // Load training data
        const trainText = await readFile(trainFile);
        console.log('Raw CSV text (first 500 chars):', trainText.substring(0, 500));
        
        rawTrainData = parseCSV(trainText);
        console.log('Parsed training data:', rawTrainData);
        
        if (rawTrainData.length === 0) {
            throw new Error('No data found in training file');
        }

        showStatus('dataStatus', `Loaded ${rawTrainData.length} training samples`, 'info');
        
        // Load test data if provided
        const testFile = document.getElementById('testFile').files[0];
        if (testFile) {
            const testText = await readFile(testFile);
            rawTestData = parseCSV(testText);
            showStatus('dataStatus', `Loaded ${rawTestData.length} test samples`, 'info');
        }

        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        // Display data preview and statistics
        displayDataPreview();
        displayDataStatistics();
        plotSurvivalCharts();
        
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus('dataStatus', `Error loading data: ${error.message}`, 'error');
    }
}

// Simple CSV parsing function
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];
    
    // Extract headers
    const headers = lines[0].split(',').map(h => h.trim());
    console.log('CSV Headers:', headers);
    
    // Parse data rows
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '') continue;
        
        const values = line.split(',');
        if (values.length !== headers.length) {
            console.warn(`Row ${i} has ${values.length} columns, expected ${headers.length}`);
            continue;
        }
        
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index].trim().replace(/"/g, ''); // Remove quotes
            
            // Convert numeric values
            if (!isNaN(value) && value !== '') {
                value = Number(value);
            } else if (value === '') {
                value = null;
            }
            
            row[header] = value;
        });
        data.push(row);
    }
    
    return data;
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

function displayDataPreview() {
    const container = document.getElementById('dataPreview');
    if (rawTrainData.length === 0) return;
    
    const previewData = rawTrainData.slice(0, 5);
    const headers = Object.keys(previewData[0]);
    
    let html = `<h3>Data Preview (First 5 Rows)</h3>
                <div class="data-preview">
                <table>
                    <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
                    <tbody>`;
    
    previewData.forEach(row => {
        html += `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`;
    });
    
    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

function displayDataStatistics() {
    const container = document.getElementById('dataStats');
    if (rawTrainData.length === 0) return;
    
    const totalSamples = rawTrainData.length;
    const features = Object.keys(rawTrainData[0]);
    
    let statsHtml = `<h3>Data Statistics</h3>
                    <p><strong>Shape:</strong> ${totalSamples} samples × ${features.length} features</p>
                    <h4>Missing Values:</h4>
                    <table>
                        <tr><th>Feature</th><th>Missing Count</th><th>Missing %</th></tr>`;
    
    features.forEach(feature => {
        const missingCount = rawTrainData.filter(row => 
            row[feature] === null || row[feature] === '' || row[feature] === undefined
        ).length;
        const missingPercent = ((missingCount / totalSamples) * 100).toFixed(2);
        statsHtml += `<tr>
            <td>${feature}</td>
            <td>${missingCount}</td>
            <td>${missingPercent}%</td>
        </tr>`;
    });
    
    statsHtml += `</table>`;
    container.innerHTML = statsHtml;
}

function plotSurvivalCharts() {
    if (rawTrainData.length === 0) return;
    
    // Survival by Sex
    const sexGroups = {};
    rawTrainData.forEach(row => {
        if (row.Sex && row.Survived !== null && row.Survived !== undefined) {
            if (!sexGroups[row.Sex]) sexGroups[row.Sex] = { survived: 0, total: 0 };
            sexGroups[row.Sex].total++;
            if (row.Survived === 1) sexGroups[row.Sex].survived++;
        }
    });
    
    const sexData = Object.entries(sexGroups).map(([sex, stats]) => ({
        x: sex,
        y: (stats.survived / stats.total) * 100
    }));
    
    // Survival by Pclass
    const classGroups = {};
    rawTrainData.forEach(row => {
        if (row.Pclass && row.Survived !== null && row.Survived !== undefined) {
            const pclass = `Class ${row.Pclass}`;
            if (!classGroups[pclass]) classGroups[pclass] = { survived: 0, total: 0 };
            classGroups[pclass].total++;
            if (row.Survived === 1) classGroups[pclass].survived++;
        }
    });
    
    const classData = Object.entries(classGroups).map(([pclass, stats]) => ({
        x: pclass,
        y: (stats.survived / stats.total) * 100
    }));
    
    // Plot using tfjs-vis
    if (sexData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival by Sex', tab: 'Data Inspection' }, 
            { values: sexData },
            { xLabel: 'Sex', yLabel: 'Survival Rate %' }
        );
    }
    
    if (classData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival by Class', tab: 'Data Inspection' }, 
            { values: classData },
            { xLabel: 'Passenger Class', yLabel: 'Survival Rate %' }
        );
    }
}

// Data preprocessing
function preprocessData() {
    try {
        showStatus('preprocessStatus', 'Preprocessing data...', 'info');
        
        // Preprocess training data
        const processedTrain = rawTrainData.map(row => preprocessRow(row, true));
        
        // Remove rows with missing target
        const validTrainData = processedTrain.filter(row => row.features && row.target !== null);
        
        // Convert to tensors
        const featureArrays = validTrainData.map(row => row.features);
        const targetArray = validTrainData.map(row => row.target);
        
        processedTrainData = {
            features: tf.tensor2d(featureArrays),
            targets: tf.tensor1d(targetArray),
            passengerIds: validTrainData.map(row => row.passengerId)
        };
        
        // Preprocess test data if available
        if (rawTestData.length > 0) {
            const processedTest = rawTestData.map(row => preprocessRow(row, false));
            const validTestData = processedTest.filter(row => row.features);
            
            const testFeatureArrays = validTestData.map(row => row.features);
            processedTestData = {
                features: tf.tensor2d(testFeatureArrays),
                passengerIds: validTestData.map(row => row.passengerId)
            };
        }
        
        showStatus('preprocessStatus', 
            `Preprocessed ${validTrainData.length} training samples. Feature dimension: ${featureArrays[0].length}`, 
            'info');
        
        displayFeatureInfo(validTrainData[0].features.length);
        document.getElementById('modelBtn').disabled = false;
        
    } catch (error) {
        console.error('Error preprocessing data:', error);
        showStatus('preprocessStatus', `Error preprocessing data: ${error.message}`, 'error');
    }
}

function preprocessRow(row, isTraining) {
    try {
        const features = [];
        const passengerId = row.PassengerId;
        
        // Handle missing values
        const age = row.Age !== null && row.Age !== '' && !isNaN(row.Age) ? row.Age : 29.7;
        const fare = row.Fare !== null && row.Fare !== '' && !isNaN(row.Fare) ? row.Fare : 14.45;
        const embarked = row.Embarked && row.Embarked !== '' ? row.Embarked : 'S';
        
        // Feature engineering
        const sibSp = row.SibSp || 0;
        const parch = row.Parch || 0;
        const familySize = sibSp + parch + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        
        // Numerical features
        features.push(age);
        features.push(fare);
        features.push(sibSp);
        features.push(parch);
        features.push(familySize);
        features.push(isAlone);
        
        // Categorical features (one-hot encoding)
        // Sex
        if (row.Sex === 'male') {
            features.push(1, 0);
        } else {
            features.push(0, 1);
        }
        
        // Pclass
        const pclass = row.Pclass || 3;
        features.push(
            pclass === 1 ? 1 : 0,
            pclass === 2 ? 1 : 0, 
            pclass === 3 ? 1 : 0
        );
        
        // Embarked
        if (embarked === 'C') {
            features.push(1, 0, 0);
        } else if (embarked === 'Q') {
            features.push(0, 1, 0);
        } else {
            features.push(0, 0, 1);
        }
        
        const target = isTraining ? (row.Survived !== null && row.Survived !== undefined ? row.Survived : null) : null;
        
        return {
            features,
            target,
            passengerId
        };
        
    } catch (error) {
        console.error('Error preprocessing row:', error, row);
        return { features: null, target: null, passengerId: null };
    }
}

function displayFeatureInfo(featureDim) {
    const container = document.getElementById('featureInfo');
    container.innerHTML = `
        <h4>Processed Features (Dimension: ${featureDim})</h4>
        <p><strong>Numerical:</strong> Age, Fare, SibSp, Parch, FamilySize, IsAlone</p>
        <p><strong>Categorical (One-hot):</strong> Sex (2), Pclass (3), Embarked (3)</p>
        <p><strong>Total Features:</strong> 6 numerical + 8 categorical = 14 features</p>
    `;
}

// Model creation
function createModel() {
    try {
        showStatus('modelStatus', 'Creating model...', 'info');
        
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [processedTrainData.features.shape[1]],
                    units: 16,
                    activation: 'relu',
                    name: 'hidden_layer'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid',
                    name: 'output_layer'
                })
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        const summaryContainer = document.getElementById('modelSummary');
        summaryContainer.innerHTML = '<h4>Model Summary</h4>';
        
        tfvis.show.modelSummary({ name: 'Model Summary', tab: 'Model' }, model);
        
        showStatus('modelStatus', 'Model created successfully!', 'info');
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('evaluateBtn').disabled = false;
        
    } catch (error) {
        console.error('Error creating model:', error);
        showStatus('modelStatus', `Error creating model: ${error.message}`, 'error');
    }
}

// Model training
async function trainModel() {
    if (!model || !processedTrainData) {
        alert('Please preprocess data and create model first');
        return;
    }
    
    try {
        isTraining = true;
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        showStatus('trainingStatus', 'Training model...', 'info');
        
        // Create train/validation split (80/20)
        const splitIndex = Math.floor(processedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = processedTrainData.features.slice(0, splitIndex);
        const trainTargets = processedTrainData.targets.slice(0, splitIndex);
        const valFeatures = processedTrainData.features.slice(splitIndex);
        const valTargets = processedTrainData.targets.slice(splitIndex);
        
        // Training callbacks
        const callbacks = tfvis.show.fitCallbacks(
            { name: 'Training Metrics', tab: 'Training' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        );
        
        // Train the model
        const history = await model.fit(trainFeatures, trainTargets, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valTargets],
            callbacks: callbacks,
            verbose: 0
        });
        
        trainingHistory = history.history;
        
        // Store validation data for metrics
        validationPredictions = model.predict(valFeatures);
        validationLabels = valTargets;
        
        showStatus('trainingStatus', 'Training completed!', 'info');
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('saveBtn').disabled = false;
        
    } catch (error) {
        console.error('Error training model:', error);
        showStatus('trainingStatus', `Error training model: ${error.message}`, 'error');
    } finally {
        isTraining = false;
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }
}

function stopTraining() {
    if (model && isTraining) {
        model.stopTraining = true;
        showStatus('trainingStatus', 'Training stopped by user', 'warning');
        document.getElementById('stopBtn').disabled = true;
    }
}

// Model evaluation and metrics
async function evaluateModel() {
    if (!model || !validationPredictions) {
        alert('Please train the model first');
        return;
    }
    
    try {
        showStatus('metricsDisplay', 'Computing metrics...', 'info');
        
        const probs = await validationPredictions.data();
        const labels = await validationLabels.data();
        
        updateMetrics(labels, probs, currentThreshold);
        showStatus('metricsDisplay', 'Evaluation completed!', 'info');
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        showStatus('metricsDisplay', `Error evaluating model: ${error.message}`, 'error');
    }
}

function updateThreshold(value) {
    currentThreshold = parseFloat(value);
    document.getElementById('thresholdValue').textContent = currentThreshold.toFixed(2);
    
    if (validationPredictions && validationLabels) {
        const probs = validationPredictions.dataSync();
        const labels = validationLabels.dataSync();
        updateMetrics(labels, probs, currentThreshold);
    }
}

function updateMetrics(labels, probabilities, threshold) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < labels.length; i++) {
        const prediction = probabilities[i] >= threshold ? 1 : 0;
        if (labels[i] === 1) {
            if (prediction === 1) tp++;
            else fn++;
        } else {
            if (prediction === 1) fp++;
            else tn++;
        }
    }
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    // Update metrics display
    const metricsHtml = `
        <div class="metrics-display">
            <h4>Classification Metrics (Threshold: ${threshold.toFixed(2)})</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>${accuracy.toFixed(4)}</td></tr>
                <tr><td>Precision</td><td>${precision.toFixed(4)}</td></tr>
                <tr><td>Recall</td><td>${recall.toFixed(4)}</td></tr>
                <tr><td>F1-Score</td><td>${f1.toFixed(4)}</td></tr>
            </table>
        </div>
    `;
    
    document.getElementById('metricsDisplay').innerHTML = metricsHtml;
    
    // Update confusion matrix table
    const confusionMatrixHtml = `
        <h4>Confusion Matrix</h4>
        <table class="confusion-matrix">
            <tr>
                <th></th>
                <th>Predicted Negative</th>
                <th>Predicted Positive</th>
            </tr>
            <tr>
                <th>Actual Negative</th>
                <td>${tn} (True Negative)</td>
                <td>${fp} (False Positive)</td>
            </tr>
            <tr>
                <th>Actual Positive</th>
                <td>${fn} (False Negative)</td>
                <td>${tp} (True Positive)</td>
            </tr>
        </table>
    `;
    
    document.getElementById('evaluationTable').innerHTML = confusionMatrixHtml;
}

// Prediction and export
async function predictTestData() {
    if (!model || !processedTestData) {
        alert('No test data available. Please load test.csv file.');
        return;
    }
    
    try {
        showStatus('predictionStatus', 'Making predictions...', 'info');
        
        // Make predictions
        testPredictions = model.predict(processedTestData.features);
        const probabilities = await testPredictions.data();
        
        // Display sample predictions
        const sampleResults = processedTestData.passengerIds.slice(0, 10).map((id, i) => ({
            passengerId: id,
            probability: probabilities[i],
            prediction: probabilities[i] >= currentThreshold ? 1 : 0
        }));
        
        let resultsHtml = `<h4>Sample Predictions (First 10)</h4>
                          <table>
                            <tr><th>PassengerId</th><th>Probability</th><th>Prediction</th></tr>`;
        
        sampleResults.forEach(result => {
            resultsHtml += `<tr>
                <td>${result.passengerId}</td>
                <td>${result.probability.toFixed(4)}</td>
                <td>${result.prediction}</td>
            </tr>`;
        });
        
        resultsHtml += `</table>
                       <p>Total test samples: ${processedTestData.passengerIds.length}</p>`;
        
        document.getElementById('predictionResults').innerHTML = resultsHtml;
        showStatus('predictionStatus', 'Predictions completed!', 'info');
        document.getElementById('exportBtn').disabled = false;
        document.getElementById('probBtn').disabled = false;
        
    } catch (error) {
        console.error('Error making predictions:', error);
        showStatus('predictionStatus', `Error making predictions: ${error.message}`, 'error');
    }
}

function exportSubmission() {
    if (!testPredictions || !processedTestData) {
        alert('No predictions available. Please predict test data first.');
        return;
    }
    
    try {
        const probabilities = testPredictions.dataSync();
        let csvContent = 'PassengerId,Survived\n';
        
        processedTestData.passengerIds.forEach((id, i) => {
            const prediction = probabilities[i] >= currentThreshold ? 1 : 0;
            csvContent += `${id},${prediction}\n`;
        });
        
        downloadCSV(csvContent, 'titanic_submission.csv');
        showStatus('predictionStatus', 'Submission CSV exported!', 'info');
        
    } catch (error) {
        console.error('Error exporting submission:', error);
        showStatus('predictionStatus', `Error exporting submission: ${error.message}`, 'error');
    }
}

function exportProbabilities() {
    if (!testPredictions || !processedTestData) {
        alert('No predictions available. Please predict test data first.');
        return;
    }
    
    try {
        const probabilities = testPredictions.dataSync();
        let csvContent = 'PassengerId,Probability\n';
        
        processedTestData.passengerIds.forEach((id, i) => {
            csvContent += `${id},${probabilities[i].toFixed(6)}\n`;
        });
        
        downloadCSV(csvContent, 'titanic_probabilities.csv');
        showStatus('predictionStatus', 'Probabilities CSV exported!', 'info');
        
    } catch (error) {
        console.error('Error exporting probabilities:', error);
        showStatus('predictionStatus', `Error exporting probabilities: ${error.message}`, 'error');
    }
}

async function saveModel() {
    if (!model) {
        alert('No model to save. Please create and train a model first.');
        return;
    }
    
    try {
        await model.save('downloads://titanic-tfjs-model');
        showStatus('predictionStatus', 'Model saved successfully!', 'info');
    } catch (error) {
        console.error('Error saving model:', error);
        showStatus('predictionStatus', `Error saving model: ${error.message}`, 'error');
    }
}

// Utility functions
function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="status ${type}">${message}</div>`;
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Initialize the application
console.log('Titanic Survival Classifier initialized');
