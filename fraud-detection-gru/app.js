import DataLoader from './data-loader.js';
import FraudDetectionModel from './module.js';

// Main application controller
class FraudDetectionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new FraudDetectionModel();
        this.isDataLoaded = false;
        this.isModelTrained = false;
        
        this.initializeEventListeners();
    }

    // Set up UI event listeners
    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('evaluateModel').addEventListener('click', () => this.evaluateModel());
        
        // Listen for training progress updates
        document.addEventListener('trainingProgress', (event) => {
            this.updateTrainingProgress(event.detail);
        });
    }

    // Load and preprocess CSV data
    async loadData() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a CSV file first');
            return;
        }

        try {
            this.showMessage('Loading CSV file...', 'info');
            
            await this.dataLoader.loadCSV(file);
            this.dataLoader.normalizeFeatures();
            this.dataLoader.createSequences(10); // 10-step sequences
            this.dataLoader.splitData(0.2); // 80/20 split
            
            this.isDataLoaded = true;
            
            // Display data information
            const dataInfo = this.dataLoader.getDataInfo();
            document.getElementById('dataInfo').innerHTML = `
                <div class="success">
                    <strong>Data loaded successfully!</strong><br>
                    Total samples: ${dataInfo.totalSamples}<br>
                    Fraud transactions: ${dataInfo.fraudCount} (${dataInfo.fraudPercentage}%)<br>
                    Non-fraud transactions: ${dataInfo.nonFraudCount}
                </div>
            `;
            
            this.showMessage('Data loaded and preprocessed successfully!', 'success');
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    // Train the GRU model
    async trainModel() {
        if (!this.isDataLoaded) {
            this.showError('Please load data first');
            return;
        }

        try {
            this.showMessage('Building and training model...', 'info');
            document.getElementById('progressLog').innerHTML = '';
            
            // Build model
            const inputShape = [
                this.dataLoader.trainSequences.shape[1], // sequence length
                this.dataLoader.trainSequences.shape[2]  // feature count
            ];
            
            this.model.buildModel(inputShape);
            
            // Display model info
            document.getElementById('modelInfo').innerHTML = `
                <div class="success">
                    <strong>Model Built Successfully</strong><br>
                    Input shape: [${inputShape.join(', ')}]<br>
                    GRU units: 32<br>
                    Architecture: GRU → Dropout → Dense(16) → Output
                </div>
            `;
            
            // Train model
            await this.model.train(
                this.dataLoader.trainSequences, 
                this.dataLoader.trainLabels, 
                20,  // epochs
                0.2  // validation split
            );
            
            this.isModelTrained = true;
            this.showMessage('Model training completed!', 'success');
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        }
    }

    // Evaluate model and visualize results
    async evaluateModel() {
        if (!this.isModelTrained) {
            this.showError('Please train the model first');
            return;
        }

        try {
            this.showMessage('Evaluating model...', 'info');
            
            // Get evaluation metrics
            const evaluation = await this.model.evaluate(
                this.dataLoader.testSequences, 
                this.dataLoader.testLabels
            );
            
            // Make predictions
            const predictions = await this.model.predict(this.dataLoader.testSequences);
            const actualLabels = await this.dataLoader.testLabels.data();
            
            // Display evaluation results
            document.getElementById('evaluationResults').innerHTML = `
                <div class="success">
                    <strong>Evaluation Results:</strong><br>
                    Test Loss: ${evaluation.loss.toFixed(4)}<br>
                    Test Accuracy: ${(evaluation.accuracy * 100).toFixed(2)}%
                </div>
            `;
            
            // Visualize predictions
            this.visualizePredictions(predictions, Array.from(actualLabels));
            
            this.showMessage('Evaluation completed!', 'success');
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    // Update training progress in UI
    updateTrainingProgress(progress) {
        const progressLog = document.getElementById('progressLog');
        const progressItem = document.createElement('div');
        progressItem.className = 'progress-item';
        progressItem.innerHTML = `
            Epoch ${progress.epoch}: 
            Loss: ${progress.loss}, 
            Accuracy: ${progress.accuracy}%,
            Val Loss: ${progress.valLoss},
            Val Accuracy: ${progress.valAccuracy}%
        `;
        progressLog.appendChild(progressItem);
        progressLog.scrollTop = progressLog.scrollHeight;
    }

    // Visualize prediction results using canvas
    visualizePredictions(predictions, actualLabels) {
        const canvas = document.getElementById('resultsCanvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Sample first 50 predictions for visualization
        const sampleSize = Math.min(50, predictions.length);
        const barWidth = width / sampleSize;
        
        let correctPredictions = 0;
        
        for (let i = 0; i < sampleSize; i++) {
            const prediction = predictions[i];
            const actual = actualLabels[i];
            const predictedClass = prediction > 0.5 ? 1 : 0;
            const isCorrect = predictedClass === actual;
            
            if (isCorrect) correctPredictions++;
            
            // Set color: green for correct, red for incorrect
            ctx.fillStyle = isCorrect ? 
                (actual === 1 ? '#4CAF50' : '#2196F3') : // Green for correct fraud, blue for correct non-fraud
                '#FF5252'; // Red for incorrect
            
            // Draw bar: height represents prediction confidence
            const barHeight = Math.abs(prediction - 0.5) * 2 * height * 0.8;
            const x = i * barWidth;
            const y = height - barHeight;
            
            ctx.fillRect(x, y, barWidth - 1, barHeight);
            
            // Add label for fraud predictions
            if (predictedClass === 1) {
                ctx.fillStyle = '#000';
                ctx.font = '10px Arial';
                ctx.fillText('F', x + barWidth/2 - 3, height - 5);
            }
        }
        
        // Add title and legend
        ctx.fillStyle = '#000';
        ctx.font = '14px Arial';
        ctx.fillText(`Prediction Results (Sample: ${sampleSize}, Correct: ${correctPredictions}/${sampleSize})`, 10, 20);
        
        ctx.fillStyle = '#4CAF50';
        ctx.fillRect(width - 200, 30, 15, 15);
        ctx.fillStyle = '#000';
        ctx.fillText('Correct Fraud', width - 180, 40);
        
        ctx.fillStyle = '#2196F3';
        ctx.fillRect(width - 200, 50, 15, 15);
        ctx.fillStyle = '#000';
        ctx.fillText('Correct Non-Fraud', width - 180, 60);
        
        ctx.fillStyle = '#FF5252';
        ctx.fillRect(width - 200, 70, 15, 15);
        ctx.fillStyle = '#000';
        ctx.fillText('Incorrect', width - 180, 80);
    }

    // Utility methods for UI messaging
    showMessage(message, type = 'info') {
        console.log(`${type.toUpperCase()}: ${message}`);
    }

    showError(message) {
        this.showMessage(message, 'error');
        document.getElementById('dataInfo').innerHTML = `<div class="error">${message}</div>`;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FraudDetectionApp();
});
