// app.js
import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class RetailForecastApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.currentData = null;
        this.entities = [];
        this.selectedEntity = null;
        this.charts = {};
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('loadModel').addEventListener('click', () => this.loadSavedModel());
        document.getElementById('entitySelect').addEventListener('change', (e) => {
            this.selectedEntity = e.target.value;
        });
    }

    async loadData() {
        const fileInput = document.getElementById('csvFile');
        if (!fileInput.files.length) {
            alert('Please select a CSV file');
            return;
        }

        try {
            this.showMessage('Loading CSV file...', 'dataInfo');
            await this.dataLoader.loadCSV(fileInput.files[0]);
            
            this.showMessage('Preprocessing data...', 'dataInfo');
            this.dataLoader.preprocessData();
            
            // Extract unique entities for selection
            this.extractEntities();
            this.populateEntitySelect();
            
            this.showMessage('Data loaded successfully! Select an entity and train model.', 'dataInfo');
            document.getElementById('trainModel').disabled = false;
            
        } catch (error) {
            this.showMessage(`Error: ${error.message}`, 'dataInfo', true);
        }
    }

    extractEntities() {
        const products = [...new Set(this.dataLoader.processedData.map(row => row.Product_id))].slice(0, 20);
        const warehouses = [...new Set(this.dataLoader.processedData.map(row => row.Warehouse))];
        
        this.entities = [
            ...products.map(id => ({ type: 'product', id, name: `Product ${id}` })),
            ...warehouses.map(id => ({ type: 'warehouse', id, name: `Warehouse ${id}` }))
        ];
    }

    populateEntitySelect() {
        const select = document.getElementById('entitySelect');
        select.innerHTML = '';
        
        this.entities.forEach(entity => {
            const option = document.createElement('option');
            option.value = `${entity.type}:${entity.id}`;
            option.textContent = entity.name;
            select.appendChild(option);
        });
        
        this.selectedEntity = select.value;
    }

    async trainModel() {
        if (!this.selectedEntity) {
            alert('Please select a product or warehouse');
            return;
        }

        try {
            const [entityType, entityId] = this.selectedEntity.split(':');
            
            this.showMessage('Preparing features...', 'trainingLog');
            const preparedData = this.dataLoader.prepareFeaturesForEntity(entityType, entityId);
            
            this.showMessage('Splitting data...', 'trainingLog');
            const { X_train, X_test, y_train, y_test } = this.dataLoader.splitData(
                preparedData.features, preparedData.labels
            );

            this.showMessage('Creating model...', 'trainingLog');
            const inputShape = [X_train.shape[1], X_train.shape[2]];
            this.model.createModel(inputShape);

            this.showMessage('Starting training...', 'trainingLog');
            document.getElementById('trainModel').disabled = true;

            await this.model.trainModel(X_train, y_train, X_test, y_test, 50, {
                onEpochEnd: (epoch, logs) => {
                    const progress = ((epoch + 1) / 50) * 100;
                    document.getElementById('trainingProgress').style.width = `${progress}%`;
                    
                    const logEntry = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}`;
                    this.showMessage(logEntry, 'trainingLog');
                },
                onTrainEnd: () => {
                    document.getElementById('trainModel').disabled = false;
                    this.evaluateAndDisplay(X_test, y_test, preparedData);
                    this.model.saveModel();
                    document.getElementById('loadModel').disabled = false;
                }
            });

            // Clean up tensors
            X_train.dispose();
            X_test.dispose();
            y_train.dispose();
            y_test.dispose();

        } catch (error) {
            this.showMessage(`Training error: ${error.message}`, 'trainingLog', true);
            document.getElementById('trainModel').disabled = false;
        }
    }

    async loadSavedModel() {
        try {
            this.showMessage('Loading saved model...', 'trainingLog');
            const success = await this.model.loadModel();
            
            if (success) {
                this.showMessage('Model loaded successfully!', 'trainingLog');
                // If we have data, run evaluation
                if (this.selectedEntity) {
                    const [entityType, entityId] = this.selectedEntity.split(':');
                    const preparedData = this.dataLoader.prepareFeaturesForEntity(entityType, entityId);
                    const { X_test, y_test } = this.dataLoader.splitData(
                        preparedData.features, preparedData.labels
                    );
                    this.evaluateAndDisplay(X_test, y_test, preparedData);
                    X_test.dispose();
                    y_test.dispose();
                }
            } else {
                this.showMessage('No saved model found. Please train a model first.', 'trainingLog', true);
            }
        } catch (error) {
            this.showMessage(`Error loading model: ${error.message}`, 'trainingLog', true);
        }
    }

    async evaluateAndDisplay(X_test, y_test, preparedData) {
        try {
            this.showMessage('Generating predictions...', 'trainingLog');
            const predictions = await this.model.predict(X_test);
            
            const metrics = this.model.evaluateModel(y_test, predictions);
            this.displayMetrics(metrics);
            
            const actualData = await y_test.array();
            const predictedData = await predictions.array();
            
            this.createForecastChart(actualData, predictedData);
            this.displayForecastResults(actualData, predictedData, preparedData);
            
            predictions.dispose();
            
        } catch (error) {
            this.showMessage(`Evaluation error: ${error.message}`, 'trainingLog', true);
        }
    }

    displayMetrics(metrics) {
        const metricsDiv = document.getElementById('metrics');
        metricsDiv.innerHTML = `
            <div class="metric-card">
                <h3>MAE</h3>
                <p>${metrics.mae.toFixed(4)}</p>
            </div>
            <div class="metric-card">
                <h3>RMSE</h3>
                <p>${metrics.rmse.toFixed(4)}</p>
            </div>
            <div class="metric-card">
                <h3>MSE</h3>
                <p>${metrics.mse.toFixed(4)}</p>
            </div>
        `;
    }

    createForecastChart(actualData, predictedData) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (this.charts.forecast) {
            this.charts.forecast.destroy();
        }

        // Use first sample for demonstration
        const sampleIndex = Math.min(0, actualData.length - 1);
        const actual = actualData[sampleIndex];
        const predicted = predictedData[sampleIndex];

        this.charts.forecast = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 7}, (_, i) => `Day ${i + 1}`),
                datasets: [
                    {
                        label: 'Actual Demand',
                        data: actual,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Predicted Demand',
                        data: predicted,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '7-Day Demand Forecast vs Actual'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Normalized Demand'
                        }
                    }
                }
            }
        });
    }

    displayForecastResults(actualData, predictedData, preparedData) {
        const resultsDiv = document.getElementById('forecastResults');
        
        // Calculate accuracy for each forecast day
        const dayAccuracies = [];
        for (let day = 0; day < 7; day++) {
            let totalError = 0;
            let count = 0;
            
            for (let sample = 0; sample < actualData.length; sample++) {
                if (actualData[sample][day] !== undefined && predictedData[sample][day] !== undefined) {
                    const error = Math.abs(actualData[sample][day] - predictedData[sample][day]);
                    totalError += error;
                    count++;
                }
            }
            
            const avgError = count > 0 ? totalError / count : 0;
            const accuracy = Math.max(0, 1 - avgError);
            dayAccuracies.push((accuracy * 100).toFixed(1));
        }
        
        resultsDiv.innerHTML = `
            <h3>Forecast Analysis</h3>
            <p>Model trained on ${preparedData.features.shape[0]} sequences</p>
            <div class="accuracy-breakdown">
                <h4>Day-by-Day Forecast Accuracy:</h4>
                <ul>
                    ${dayAccuracies.map((acc, i) => `<li>Day ${i + 1}: ${acc}% accuracy</li>`).join('')}
                </ul>
            </div>
        `;
    }

    showMessage(message, elementId, isError = false) {
        const element = document.getElementById(elementId);
        element.innerHTML = message;
        element.style.color = isError ? 'red' : 'black';
        
        if (elementId === 'trainingLog') {
            // Keep log history
            const timestamp = new Date().toLocaleTimeString();
            element.innerHTML += `<br><small>${timestamp}: ${message}</small>`;
            element.scrollTop = element.scrollHeight;
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RetailForecastApp();
});
