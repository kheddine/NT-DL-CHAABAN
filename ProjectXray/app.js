import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class RetailForecastingApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.isDataLoaded = false;
        this.isModelTrained = false;
        this.predictions = {};
        this.metrics = {};
        
        this.initializeEventListeners();
        this.checkForSavedModel();
    }

    initializeEventListeners() {
        // File upload
        document.getElementById('csvFile').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('loadData').disabled = false;
            }
        });

        // Data loading
        document.getElementById('loadData').addEventListener('click', () => {
            this.loadData();
        });

        // Model training
        document.getElementById('trainModel').addEventListener('click', () => {
            this.trainModel();
        });

        // Product selection
        document.getElementById('productSelect').addEventListener('change', (e) => {
            this.displayProductForecast(e.target.value);
        });

        // Download results
        document.getElementById('downloadResults').addEventListener('click', () => {
            this.downloadResults();
        });

        // Training progress updates
        window.addEventListener('trainingProgress', (e) => {
            this.updateTrainingProgress(e.detail);
        });
    }

    async checkForSavedModel() {
        const loaded = await this.model.loadModel();
        if (loaded) {
            this.isModelTrained = true;
            document.getElementById('trainModel').disabled = true;
            document.getElementById('trainModel').textContent = 'Model Loaded from Storage';
            this.showNotification('Model loaded from browser storage', 'success');
        }
    }

    async loadData() {
        const file = document.getElementById('csvFile').files[0];
        if (!file) {
            this.showNotification('Please select a CSV file', 'error');
            return;
        }

        try {
            this.showNotification('Loading CSV file...', 'info');
            await this.dataLoader.loadCSV(file);
            
            this.showNotification('Preprocessing data...', 'info');
            this.dataLoader.preprocessData();
            
            this.isDataLoaded = true;
            document.getElementById('trainModel').disabled = false;
            
            const productCount = this.dataLoader.getAllProducts().length;
            document.getElementById('dataInfo').innerHTML = 
                `<div class="alert alert-success">✅ Loaded ${productCount} products successfully</div>`;
                
            this.showNotification('Data loaded and processed successfully', 'success');
            
        } catch (error) {
            this.showNotification(`Error loading data: ${error.message}`, 'error');
            console.error(error);
        }
    }

    async trainModel() {
        if (!this.isDataLoaded) {
            this.showNotification('Please load data first', 'error');
            return;
        }

        try {
            const epochs = parseInt(document.getElementById('epochs').value);
            const batchSize = parseInt(document.getElementById('batchSize').value);
            const learningRate = parseFloat(document.getElementById('learningRate').value);

            document.getElementById('trainModel').disabled = true;
            document.getElementById('trainingProgress').innerHTML = 
                '<div class="progress"><div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 0%"></div></div>';

            // Prepare data
            const { trainData, testData } = this.dataLoader.prepareTrainTestSplit();
            
            // Create and train model
            const inputShape = [30, this.calculateFeatureCount()];
            this.model.createModel(inputShape, learningRate);
            
            await this.model.trainModel(trainData.sequences, trainData.targets, epochs, batchSize);
            
            // Make predictions and calculate metrics
            await this.generatePredictionsAndMetrics(testData);
            
            // Save model
            await this.model.saveModel();
            
            this.isModelTrained = true;
            this.updateUIAfterTraining();
            this.showNotification('Model training completed successfully', 'success');
            
        } catch (error) {
            this.showNotification(`Training failed: ${error.message}`, 'error');
            console.error(error);
            document.getElementById('trainModel').disabled = false;
        }
    }

    calculateFeatureCount() {
        const featureInfo = this.dataLoader.getFeatureInfo();
        return 1 + // Order_Demand
               featureInfo.categories.Warehouse.length +
               featureInfo.categories.Product_Category.length +
               4 + // Open, Promo, StateHoliday, SchoolHoliday
               1 + // Petrol_price
               2;  // Day of week (sine/cosine)
    }

    updateTrainingProgress(progress) {
        const progressBar = document.querySelector('#trainingProgress .progress-bar');
        const percent = (progress.epoch / progress.totalEpochs) * 100;
        
        progressBar.style.width = `${percent}%`;
        progressBar.textContent = `Epoch ${progress.epoch}/${progress.totalEpochs} - Loss: ${progress.loss?.toFixed(4)}`;
    }

    async generatePredictionsAndMetrics(testData) {
        this.predictions = {};
        this.metrics = {
            global: { mae: 0, rmse: 0, mape: 0, directionalAccuracy: 0 },
            products: {}
        };

        const predictions = await this.model.predict(testData.sequences);
        const featureInfo = this.dataLoader.getFeatureInfo();

        // Group predictions by product
        testData.productMap.forEach((productId, index) => {
            if (!this.predictions[productId]) {
                this.predictions[productId] = [];
            }
            this.predictions[productId].push({
                actual: testData.targets[index].map(val => 
                    this.dataLoader.denormalizeValue(val, 'Order_Demand')
                ),
                predicted: predictions[index].map(val => 
                    this.dataLoader.denormalizeValue(val, 'Order_Demand')
                )
            });
        });

        // Calculate metrics for each product
        Object.keys(this.predictions).forEach(productId => {
            const productMetrics = this.calculateProductMetrics(this.predictions[productId]);
            this.metrics.products[productId] = productMetrics;
        });

        // Calculate global metrics
        this.calculateGlobalMetrics();
    }

    calculateProductMetrics(productPredictions) {
        let totalMae = 0;
        let totalMse = 0;
        let totalMape = 0;
        let totalDirectional = 0;
        let totalPoints = 0;

        productPredictions.forEach(prediction => {
            for (let i = 0; i < 7; i++) {
                const actual = prediction.actual[i];
                const predicted = prediction.predicted[i];
                const error = Math.abs(actual - predicted);
                
                totalMae += error;
                totalMse += error * error;
                
                if (actual > 0) {
                    totalMape += (error / actual) * 100;
                }
                
                // Directional accuracy (compare with previous day in sequence)
                if (i > 0) {
                    const actualDir = Math.sign(actual - prediction.actual[i-1]);
                    const predictedDir = Math.sign(predicted - prediction.predicted[i-1]);
                    if (actualDir === predictedDir) totalDirectional++;
                }
                
                totalPoints++;
            }
        });

        return {
            mae: totalMae / totalPoints,
            rmse: Math.sqrt(totalMse / totalPoints),
            mape: totalMape / totalPoints,
            directionalAccuracy: (totalDirectional / (totalPoints - productPredictions.length)) * 100
        };
    }

    calculateGlobalMetrics() {
        const productMetrics = Object.values(this.metrics.products);
        this.metrics.global = {
            mae: productMetrics.reduce((sum, m) => sum + m.mae, 0) / productMetrics.length,
            rmse: productMetrics.reduce((sum, m) => sum + m.rmse, 0) / productMetrics.length,
            mape: productMetrics.reduce((sum, m) => sum + m.mape, 0) / productMetrics.length,
            directionalAccuracy: productMetrics.reduce((sum, m) => sum + m.directionalAccuracy, 0) / productMetrics.length
        };
    }

    updateUIAfterTraining() {
        // Update global metrics
        this.displayGlobalMetrics();
        
        // Populate product dropdown
        this.populateProductDropdown();
        
        // Create accuracy chart
        this.createAccuracyChart();
        
        // Enable download button
        document.getElementById('downloadResults').disabled = false;
    }

    displayGlobalMetrics() {
        const global = this.metrics.global;
        document.getElementById('globalMetrics').innerHTML = `
            <div class="col-md-3">
                <div class="card text-white bg-primary">
                    <div class="card-body">
                        <h5 class="card-title">MAE</h5>
                        <h2 class="card-text">${global.mae.toFixed(2)}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-success">
                    <div class="card-body">
                        <h5 class="card-title">RMSE</h5>
                        <h2 class="card-text">${global.rmse.toFixed(2)}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-warning">
                    <div class="card-body">
                        <h5 class="card-title">MAPE</h5>
                        <h2 class="card-text">${global.mape.toFixed(1)}%</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-info">
                    <div class="card-body">
                        <h5 class="card-title">Directional Acc</h5>
                        <h2 class="card-text">${global.directionalAccuracy.toFixed(1)}%</h2>
                    </div>
                </div>
            </div>
        `;
    }

    populateProductDropdown() {
        const select = document.getElementById('productSelect');
        select.disabled = false;
        select.innerHTML = '<option value="">Select a product...</option>';
        
        // Sort products by MAPE (accuracy)
        const sortedProducts = Object.keys(this.metrics.products).sort((a, b) => 
            this.metrics.products[a].mape - this.metrics.products[b].mape
        );
        
        sortedProducts.forEach(productId => {
            const productData = this.dataLoader.getProductData(productId);
            const mape = this.metrics.products[productId].mape;
            const option = document.createElement('option');
            option.value = productId;
            option.textContent = `${productData.productInfo.code} (MAPE: ${mape.toFixed(1)}%)`;
            select.appendChild(option);
        });
    }

    createAccuracyChart() {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        // Get top 20 products by accuracy (lowest MAPE)
        const sortedProducts = Object.keys(this.metrics.products)
            .sort((a, b) => this.metrics.products[a].mape - this.metrics.products[b].mape)
            .slice(0, 20);
        
        const labels = sortedProducts.map(id => {
            const data = this.dataLoader.getProductData(id);
            return data.productInfo.code;
        });
        
        const data = sortedProducts.map(id => this.metrics.products[id].mape);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'MAPE % (Lower is Better)',
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Mean Absolute Percentage Error (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Products'
                        }
                    }
                }
            }
        });
    }

    displayProductForecast(productId) {
        if (!productId) return;
        
        const productData = this.dataLoader.getProductData(productId);
        const predictions = this.predictions[productId];
        const metrics = this.metrics.products[productId];
        
        if (!predictions || predictions.length === 0) return;
        
        // Use the last prediction for display
        const lastPrediction = predictions[predictions.length - 1];
        
        // Update forecast chart
        this.updateForecastChart(lastPrediction, productData.productInfo.code);
        
        // Update forecast table
        this.updateForecastTable(lastPrediction);
    }

    updateForecastChart(prediction, productCode) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        const days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
        
        // Destroy existing chart if it exists
        if (this.forecastChartInstance) {
            this.forecastChartInstance.destroy();
        }
        
        this.forecastChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: days,
                datasets: [
                    {
                        label: 'Actual Demand',
                        data: prediction.actual,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    },
                    {
                        label: 'Predicted Demand',
                        data: prediction.predicted,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `7-Day Forecast - ${productCode}`
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Order Demand'
                        }
                    }
                }
            }
        });
    }

    updateForecastTable(prediction) {
        const tbody = document.querySelector('#forecastTable tbody');
        tbody.innerHTML = '';
        
        for (let i = 0; i < 7; i++) {
            const actual = prediction.actual[i];
            const predicted = prediction.predicted[i];
            const error = actual - predicted;
            const errorPercent = (Math.abs(error) / actual) * 100;
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>Day ${i + 1}</td>
                <td>${actual.toFixed(2)}</td>
                <td>${predicted.toFixed(2)}</td>
                <td class="${error >= 0 ? 'text-success' : 'text-danger'}">${error.toFixed(2)}</td>
                <td class="${errorPercent < 10 ? 'text-success' : errorPercent < 20 ? 'text-warning' : 'text-danger'}">${errorPercent.toFixed(1)}%</td>
            `;
            tbody.appendChild(row);
        }
    }

    downloadResults() {
        let csvContent = 'Product_Code,Product_ID,Day,Actual,Predicted,Error,Error_Percent\n';
        
        Object.keys(this.predictions).forEach(productId => {
            const productData = this.dataLoader.getProductData(productId);
            const predictions = this.predictions[productId];
            
            // Use last prediction for each product
            const lastPrediction = predictions[predictions.length - 1];
            
            for (let i = 0; i < 7; i++) {
                const actual = lastPrediction.actual[i];
                const predicted = lastPrediction.predicted[i];
                const error = actual - predicted;
                const errorPercent = (Math.abs(error) / actual) * 100;
                
                csvContent += `"${productData.productInfo.code}","${productId}",${i + 1},${actual.toFixed(2)},${predicted.toFixed(2)},${error.toFixed(2)},${errorPercent.toFixed(2)}\n`;
            }
        });
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'retail_forecasts.csv';
        a.click();
        window.URL.revokeObjectURL(url);
    }

    showNotification(message, type = 'info') {
        // Simple notification implementation
        const alertClass = type === 'error' ? 'alert-danger' : 
                          type === 'success' ? 'alert-success' : 'alert-info';
        
        const notification = document.createElement('div');
        notification.className = `alert ${alertClass} alert-dismissible fade show`;
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Add to top of container
        const container = document.querySelector('.container');
        container.insertBefore(notification, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RetailForecastingApp();
});
