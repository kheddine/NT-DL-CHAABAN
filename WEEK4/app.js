import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/+esm';
import { StockDataLoader } from './data-loader.js';
import { GRUStockPredictor } from './gru.js';

export class StockPredictionApp {
    constructor() {
        this.dataLoader = new StockDataLoader();
        this.model = new GRUStockPredictor();
        this.isTraining = false;
        this.results = null;
        
        this.initializeUI();
    }

    initializeUI() {
        // Create main UI elements
        this.createFileUpload();
        this.createTrainingControls();
        this.createResultsContainer();
    }

    createFileUpload() {
        const container = document.getElementById('fileUploadContainer') || document.body;
        
        const fileSection = document.createElement('div');
        fileSection.innerHTML = `
            <h3>1. Upload Stock Data CSV</h3>
            <input type="file" id="csvFile" accept=".csv" />
            <div id="fileInfo"></div>
        `;
        container.appendChild(fileSection);

        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });
    }

    createTrainingControls() {
        const container = document.getElementById('trainingControls') || document.body;
        
        const controls = document.createElement('div');
        controls.innerHTML = `
            <h3>2. Train Model</h3>
            <button id="trainBtn" disabled>Start Training</button>
            <div id="trainingProgress">
                <div id="progressBar"></div>
                <div id="progressText"></div>
            </div>
        `;
        container.appendChild(controls);

        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });
    }

    createResultsContainer() {
        const container = document.getElementById('resultsContainer') || document.body;
        
        const results = document.createElement('div');
        results.innerHTML = `
            <h3>3. Results</h3>
            <div id="accuracyChart"></div>
            <div id="predictionTimelines"></div>
            <div id="confusionMatrices"></div>
        `;
        container.appendChild(results);
    }

    async handleFileUpload(file) {
        if (!file) return;

        try {
            document.getElementById('fileInfo').textContent = `Loading ${file.name}...`;
            await this.dataLoader.loadCSV(file);
            document.getElementById('fileInfo').textContent = 
                `Loaded ${this.dataLoader.symbols.length} stocks with ${this.dataLoader.X_train.shape[0]} training samples`;
            
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            document.getElementById('fileInfo').textContent = `Error: ${error.message}`;
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        const trainBtn = document.getElementById('trainBtn');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        trainBtn.disabled = true;
        trainBtn.textContent = 'Training...';
        
        try {
            await this.model.train(
                this.dataLoader.X_train, 
                this.dataLoader.y_train,
                this.dataLoader.X_test,
                this.dataLoader.y_test,
                30,  // epochs
                32   // batchSize
            );
            
            // Evaluate model
            const predictions = await this.model.predict(this.dataLoader.X_test);
            this.results = this.model.evaluatePerStock(
                this.dataLoader.y_test, 
                predictions, 
                this.dataLoader.getSymbols()
            );
            
            this.displayResults();
            
        } catch (error) {
            console.error('Training failed:', error);
            progressText.textContent = `Training failed: ${error.message}`;
        } finally {
            this.isTraining = false;
            trainBtn.disabled = false;
            trainBtn.textContent = 'Start Training';
            predictions?.dispose();
        }
    }

    displayResults() {
        this.displayAccuracyChart();
        this.displayPredictionTimelines();
    }

    displayAccuracyChart() {
        const container = document.getElementById('accuracyChart');
        const symbols = Object.keys(this.results);
        
        // Sort by accuracy
        const sortedSymbols = symbols.sort((a, b) => 
            this.results[b].accuracy - this.results[a].accuracy
        );

        const chartHtml = `
            <h4>Stock Prediction Accuracy (Sorted)</h4>
            <div style="display: flex; flex-direction: column; gap: 5px;">
                ${sortedSymbols.map(symbol => {
                    const accuracy = this.results[symbol].accuracy;
                    const width = (accuracy * 100) + '%';
                    const color = accuracy > 0.6 ? '#4CAF50' : accuracy > 0.5 ? '#FFC107' : '#F44336';
                    
                    return `
                        <div style="display: flex; align-items: center;">
                            <span style="width: 80px;">${symbol}</span>
                            <div style="flex: 1; background: #eee; height: 20px; margin: 0 10px;">
                                <div style="width: ${width}; background: ${color}; height: 100%; 
                                    display: flex; align-items: center; justify-content: flex-end; 
                                    padding-right: 5px; color: white; font-size: 12px;">
                                    ${(accuracy * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
        
        container.innerHTML = chartHtml;
    }

    displayPredictionTimelines() {
        const container = document.getElementById('predictionTimelines');
        const symbols = this.dataLoader.getSymbols();
        
        // Show first 50 predictions for timeline visualization
        const sampleCount = Math.min(50, this.results[symbols[0]].predictions.length);
        
        const timelinesHtml = symbols.map(symbol => {
            const results = this.results[symbol];
            const accuracy = (results.accuracy * 100).toFixed(1);
            
            return `
                <div style="margin: 20px 0;">
                    <h5>${symbol} (Accuracy: ${accuracy}%)</h5>
                    <div style="display: flex; flex-wrap: wrap; gap: 2px;">
                        ${results.predictions.slice(0, sampleCount).map((pred, idx) => {
                            const correct = pred === results.actuals[idx];
                            return `<div style="width: 8px; height: 8px; background: ${correct ? '#4CAF50' : '#F44336'}; 
                                    border-radius: 1px;" title="Pred: ${pred}, Actual: ${results.actuals[idx]}"></div>`;
                        }).join('')}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 5px;">
                        Green: Correct, Red: Wrong (showing first ${sampleCount} predictions)
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `
            <h4>Prediction Timelines</h4>
            ${timelinesHtml}
        `;
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.stockApp = new StockPredictionApp();
});
