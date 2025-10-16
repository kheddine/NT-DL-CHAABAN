import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

export class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.isTraining = false;
        this.testPredictions = null;
        this.testLabels = null;
        this.symbols = [];
        
        this.initializeUI();
        this.setupEventListeners();
    }

    initializeUI() {
        // Ensure required DOM elements exist
        if (!document.getElementById('csvFile') || !document.getElementById('trainBtn')) {
            console.error('Required DOM elements not found');
            return;
        }
    }

    setupEventListeners() {
        // File input handler
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Train button handler
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });

        // Listen for training progress
        window.addEventListener('trainingProgress', (e) => {
            this.updateTrainingProgress(e.detail);
        });
    }

    async handleFileUpload(file) {
        if (!file) return;
        
        try {
            this.updateStatus('Loading CSV file...');
            await this.dataLoader.loadCSV(file);
            this.dataLoader.createSamples();
            
            const data = this.dataLoader.getData();
            this.symbols = data.symbols;
            
            this.updateStatus(`Data loaded: ${this.symbols.length} stocks, ${data.X_train.shape[0]} training samples, ${data.X_test.shape[0]} test samples`);
            document.getElementById('trainBtn').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Error loading file: ${error.message}`);
            console.error('File loading error:', error);
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            document.getElementById('trainBtn').disabled = true;
            this.updateStatus('Building model...');
            
            // Clear previous model
            this.model.dispose();
            this.model.buildModel();
            
            const data = this.dataLoader.getData();
            this.updateStatus('Starting training...');
            
            await this.model.train(data.X_train, data.y_train, data.X_test, data.y_test, 25, 16); // Reduced for speed
            
            this.updateStatus('Training completed. Evaluating model...');
            await this.evaluateModel();
            
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`);
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            document.getElementById('trainBtn').disabled = false;
        }
    }

    async evaluateModel() {
        const data = this.dataLoader.getData();
        
        try {
            // Get predictions
            this.testPredictions = await this.model.predict(data.X_test);
            this.testLabels = data.y_test;
            
            // Compute overall accuracy
            const evaluation = this.model.evaluate(data.X_test, data.y_test);
            const overallAccuracy = evaluation[1].dataSync()[0];
            
            // Compute per-stock accuracy
            const perStockAccuracy = this.model.computePerStockAccuracy(
                this.testPredictions, this.testLabels, this.symbols
            );
            
            this.displayResults(overallAccuracy, perStockAccuracy);
            this.createVisualizations(perStockAccuracy);
            
            // Clean up
            tf.dispose(evaluation);
            
        } catch (error) {
            this.updateStatus(`Evaluation error: ${error.message}`);
            console.error('Evaluation error:', error);
        }
    }

    displayResults(overallAccuracy, perStockAccuracy) {
        const resultsDiv = document.getElementById('results');
        if (!resultsDiv) return;
        
        resultsDiv.innerHTML = `
            <h3>Model Results</h3>
            <p><strong>Overall Test Accuracy:</strong> ${(overallAccuracy * 100).toFixed(2)}%</p>
        `;
        
        // Create sorted accuracy list
        const sortedAccuracies = Object.entries(perStockAccuracy)
            .sort(([, accA], [, accB]) => accB - accA);
        
        let accuracyHTML = '<h4>Per-Stock Accuracy (Sorted)</h4><ul>';
        sortedAccuracies.forEach(([symbol, accuracy]) => {
            const accuracyPercent = (accuracy * 100).toFixed(2);
            accuracyHTML += `<li>${symbol}: ${accuracyPercent}%</li>`;
        });
        accuracyHTML += '</ul>';
        
        resultsDiv.innerHTML += accuracyHTML;
    }

    createVisualizations(perStockAccuracy) {
        this.createAccuracyBarChart(perStockAccuracy);
        this.createPredictionTimelines();
    }

    createAccuracyBarChart(perStockAccuracy) {
        const canvas = document.getElementById('accuracyChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Sort stocks by accuracy
        const sortedStocks = Object.entries(perStockAccuracy)
            .sort(([, accA], [, accB]) => accB - accA);
        
        const margin = { top: 40, right: 20, bottom: 60, left: 80 };
        const width = canvas.width - margin.left - margin.right;
        const height = canvas.height - margin.top - margin.bottom;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw title
        ctx.fillStyle = '#333';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Stock Prediction Accuracy Ranking', canvas.width / 2, 20);
        
        // Draw bars
        const barHeight = height / sortedStocks.length;
        const maxAccuracy = Math.max(...Object.values(perStockAccuracy)) || 1;
        
        sortedStocks.forEach(([symbol, accuracy], index) => {
            const barWidth = (accuracy / maxAccuracy) * width;
            const y = margin.top + index * barHeight;
            
            // Bar color based on accuracy
            const color = accuracy > 0.6 ? '#4CAF50' : accuracy > 0.5 ? '#FFC107' : '#F44336';
            
            // Draw bar
            ctx.fillStyle = color;
            ctx.fillRect(margin.left, y, barWidth, barHeight - 5);
            
            // Draw stock label
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.fillText(symbol, margin.left - 5, y + barHeight / 2 + 4);
            
            // Draw accuracy percentage
            ctx.textAlign = 'left';
            ctx.fillText(`${(accuracy * 100).toFixed(1)}%`, margin.left + barWidth + 5, y + barHeight / 2 + 4);
        });
        
        // Draw axis labels
        ctx.fillStyle = '#666';
        ctx.textAlign = 'center';
        ctx.fillText('Accuracy', canvas.width / 2, canvas.height - 10);
    }

    createPredictionTimelines() {
        const container = document.getElementById('timelineContainer');
        if (!container || !this.testPredictions || !this.testLabels) return;
        
        container.innerHTML = '<h4>Prediction Timelines (Sample of Test Data)</h4>';
        
        // Get a subset of test samples for visualization
        const predData = this.testPredictions.arraySync();
        const trueData = this.testLabels.arraySync();
        
        // Show timeline for first 3 stocks, first 20 test samples
        const stocksToShow = this.symbols.slice(0, Math.min(3, this.symbols.length));
        const samplesToShow = Math.min(20, predData.length);
        
        stocksToShow.forEach(symbol => {
            const stockIndex = this.symbols.indexOf(symbol);
            const timelineDiv = document.createElement('div');
            timelineDiv.className = 'timeline';
            timelineDiv.innerHTML = `<h5>${symbol} Predictions</h5>`;
            
            const canvas = document.createElement('canvas');
            canvas.width = 800;
            canvas.height = 100;
            timelineDiv.appendChild(canvas);
            container.appendChild(timelineDiv);
            
            this.drawStockTimeline(canvas, symbol, stockIndex, predData, trueData, samplesToShow);
        });
    }

    drawStockTimeline(canvas, symbol, stockIndex, predData, trueData, samplesToShow) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const pointWidth = width / samplesToShow;
        
        ctx.clearRect(0, 0, width, height);
        
        for (let sampleIdx = 0; sampleIdx < samplesToShow; sampleIdx++) {
            for (let day = 0; day < 3; day++) {
                const predIdx = stockIndex * 3 + day;
                const predicted = predData[sampleIdx][predIdx] > 0.5 ? 1 : 0;
                const actual = trueData[sampleIdx][predIdx];
                
                const x = sampleIdx * pointWidth + (day * pointWidth / 3);
                const y = height / 2;
                const radius = 4;
                
                // Color: green for correct, red for wrong
                ctx.fillStyle = predicted === actual ? '#4CAF50' : '#F44336';
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw day indicator for first sample only
                if (sampleIdx === 0) {
                    ctx.fillStyle = '#666';
                    ctx.font = '8px Arial';
                    ctx.fillText(`D${day + 1}`, x - 5, y + 15);
                }
            }
        }
        
        // Draw legend
        ctx.fillStyle = '#333';
        ctx.font = '10px Arial';
        ctx.fillText('● Correct', 10, 15);
        ctx.fillStyle = '#F44336';
        ctx.fillText('● Wrong', 80, 15);
    }

    updateTrainingProgress(detail) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            const progress = (detail.epoch / detail.epochs) * 100;
            progressBar.style.width = `${progress}%`;
        }
        
        this.updateStatus(`Training: Epoch ${detail.epoch}/${detail.epochs} - Loss: ${detail.loss.toFixed(4)}, Acc: ${detail.accuracy.toFixed(4)}`);
    }

    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
        console.log(message);
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
        if (this.testPredictions && !this.testPredictions.isDisposed) {
            this.testPredictions.dispose();
        }
        if (this.testLabels && !this.testLabels.isDisposed) {
            this.testLabels.dispose();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StockPredictionApp();
});
