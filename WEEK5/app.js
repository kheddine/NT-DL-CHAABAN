import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
import { DataLoader } from './data-loader.js';
import { RNNModel } from './rnn.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new RNNModel();
        this.isTraining = false;
        this.symbols = [];
        this.predictions = null;
        this.initializeUI();
    }

    initializeUI() {
        // File input handler
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Train button handler
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });

        // Initialize charts
        this.initializeCharts();
    }

    async handleFileUpload(file) {
        if (!file) return;
        
        try {
            document.getElementById('status').textContent = 'Loading CSV file...';
            await this.dataLoader.loadCSV(file);
            this.dataLoader.normalizeData();
            this.dataLoader.createSamples();
            this.symbols = this.dataLoader.getSymbols();
            
            document.getElementById('status').textContent = 'Data loaded successfully!';
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            document.getElementById('status').textContent = `Error: ${error.message}`;
            console.error(error);
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('status').textContent = 'Training model...';
        
        try {
            // Build model
            this.model.buildModel();
            
            // Set up epoch callback for UI updates
            this.model.onEpochEnd = (epoch, logs) => {
                document.getElementById('status').textContent = 
                    `Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc.toFixed(4)}`;
            };

            // Train model
            await this.model.train(
                this.dataLoader.X_train, 
                this.dataLoader.y_train, 
                this.dataLoader.X_test, 
                this.dataLoader.y_test,
                30,  // epochs
                32   // batchSize
            );

            // Make predictions
            this.predictions = await this.model.predict(this.dataLoader.X_test);
            
            // Evaluate and visualize
            this.evaluateAndVisualize();
            
            document.getElementById('status').textContent = 'Training completed!';
            
        } catch (error) {
            document.getElementById('status').textContent = `Training error: ${error.message}`;
            console.error(error);
        } finally {
            this.isTraining = false;
            document.getElementById('trainBtn').disabled = false;
        }
    }

    evaluateAndVisualize() {
        // Compute stock accuracies
        const stockAccuracies = this.model.computeStockAccuracies(
            this.predictions, 
            this.dataLoader.y_test, 
            this.symbols
        );

        // Sort stocks by accuracy
        const sortedStocks = Object.entries(stockAccuracies)
            .sort(([,a], [,b]) => b - a)
            .map(([symbol, accuracy]) => ({ symbol, accuracy }));

        // Update accuracy chart
        this.updateAccuracyChart(sortedStocks);
        
        // Update prediction timeline
        this.updatePredictionTimeline(sortedStocks);
    }

    initializeCharts() {
        // Accuracy chart canvas
        this.accuracyChart = new Chart(
            document.getElementById('accuracyChart'),
            {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Prediction Accuracy',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Stock Prediction Accuracy (Sorted)'
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 1.0,
                            title: {
                                display: true,
                                text: 'Accuracy'
                            }
                        }
                    }
                }
            }
        );

        // Timeline chart
        this.timelineChart = new Chart(
            document.getElementById('timelineChart'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Prediction Results Timeline'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Test Samples'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Stock'
                            }
                        }
                    }
                }
            }
        );
    }

    updateAccuracyChart(sortedStocks) {
        this.accuracyChart.data.labels = sortedStocks.map(item => item.symbol);
        this.accuracyChart.data.datasets[0].data = sortedStocks.map(item => item.accuracy);
        this.accuracyChart.update();
    }

    updatePredictionTimeline(sortedStocks) {
        // For demonstration, show accuracy trend across test samples for top 3 stocks
        const topStocks = sortedStocks.slice(0, 3);
        const predArray = this.predictions.arraySync();
        const testArray = this.dataLoader.y_test.arraySync();
        
        const datasets = topStocks.map((stock, idx) => {
            const stockIdx = this.symbols.indexOf(stock.symbol);
            const accuracies = [];
            
            // Calculate rolling accuracy for visualization
            for (let sample = 0; sample < predArray.length; sample += 5) {
                let correct = 0;
                let total = 0;
                
                for (let i = sample; i < Math.min(sample + 5, predArray.length); i++) {
                    for (let dayOffset = 0; dayOffset < 3; dayOffset++) {
                        const outputIdx = stockIdx + dayOffset * this.symbols.length;
                        const pred = predArray[i][outputIdx] > 0.5 ? 1 : 0;
                        const actual = testArray[i][outputIdx];
                        
                        if (pred === actual) correct++;
                        total++;
                    }
                }
                accuracies.push(total > 0 ? correct / total : 0);
            }
            
            return {
                label: stock.symbol,
                data: accuracies,
                borderColor: `hsl(${idx * 120}, 70%, 50%)`,
                backgroundColor: `hsla(${idx * 120}, 70%, 50%, 0.1)`,
                tension: 0.3,
                fill: false
            };
        });

        this.timelineChart.data.labels = Array.from({length: datasets[0].data.length}, (_, i) => `Sample ${i * 5}`);
        this.timelineChart.data.datasets = datasets;
        this.timelineChart.update();
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
        if (this.predictions) {
            this.predictions.dispose();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.stockApp = new StockPredictionApp();
});
