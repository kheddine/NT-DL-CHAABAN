// app.js
class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.trainingData = null;
        this.evaluationResults = null;
        
        this.initializeUI();
        this.setupEventListeners();
    }

    // Initialize UI components
    initializeUI() {
        this.csvFileInput = document.getElementById('csvFile');
        this.trainBtn = document.getElementById('trainBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.trainingProgress = document.getElementById('trainingProgress');
        this.statusElement = document.getElementById('status');
        
        this.accuracyChart = null;
        this.timelineChart = null;
    }

    // Set up event listeners
    setupEventListeners() {
        this.csvFileInput.addEventListener('change', (e) => this.onCSVUpload(e));
        this.trainBtn.addEventListener('click', () => this.onTrain());
        this.predictBtn.addEventListener('click', () => this.onPredict());
    }

    // Handle CSV file upload
    async onCSVUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.updateStatus('Loading and processing CSV data...');
        this.trainBtn.disabled = true;
        this.predictBtn.disabled = true;

        try {
            this.trainingData = await this.dataLoader.processData(file);
            this.updateStatus(`Data loaded successfully! ${this.trainingData.X_train.shape[0]} training samples prepared.`);
            this.trainBtn.disabled = false;
        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, true);
            console.error('CSV processing error:', error);
        }
    }

    // Handle model training
    async onTrain() {
        if (!this.trainingData) {
            this.updateStatus('Please load data first!', true);
            return;
        }

        this.updateStatus('Initializing model...');
        this.trainBtn.disabled = true;
        this.predictBtn.disabled = true;
        
        const inputShape = [this.trainingData.X_train.shape[1], this.trainingData.X_train.shape[2]];
        const outputSize = this.trainingData.symbols.length;
        
        this.model = new RNN_GRU_Model(inputShape, outputSize);
        this.model.createModel();
        
        this.updateStatus('Starting model training...');
        this.trainingProgress.classList.remove('hidden');
        this.trainingProgress.value = 0;

        try {
            await this.model.train(
                this.trainingData.X_train,
                this.trainingData.y_train,
                this.trainingData.X_val,
                this.trainingData.y_val,
                {
                    onEpochEnd: (epochInfo) => {
                        const progress = Math.round(epochInfo.progress);
                        this.trainingProgress.value = progress;
                        this.updateStatus(
                            `Epoch ${epochInfo.epoch}/50 - Loss: ${epochInfo.loss.toFixed(4)} - Acc: ${(epochInfo.accuracy * 100).toFixed(2)}%` +
                            (epochInfo.valLoss ? ` - Val Loss: ${epochInfo.valLoss.toFixed(4)} - Val Acc: ${(epochInfo.valAccuracy * 100).toFixed(2)}%` : '')
                        );
                    },
                    onTrainEnd: (history) => {
                        this.trainingProgress.classList.add('hidden');
                        const finalEpoch = history[history.length - 1];
                        this.updateStatus(
                            `Training completed! Final accuracy: ${(finalEpoch.accuracy * 100).toFixed(2)}%` +
                            (finalEpoch.val_accuracy ? `, Validation accuracy: ${(finalEpoch.val_accuracy * 100).toFixed(2)}%` : '')
                        );
                        this.predictBtn.disabled = false;
                        
                        // Save model weights
                        this.model.saveModel();
                    }
                }
            );
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`, true);
            console.error('Training error:', error);
            this.trainBtn.disabled = false;
        }
    }

    // Handle predictions
    async onPredict() {
        if (!this.model || !this.trainingData) {
            this.updateStatus('Please train the model first!', true);
            return;
        }

        this.updateStatus('Making predictions...');
        this.predictBtn.disabled = true;

        try {
            this.evaluationResults = await this.model.evaluate(
                this.trainingData.X_test,
                this.trainingData.y_test,
                this.trainingData.symbols
            );

            this.updateStatus('Predictions completed! Rendering charts...');
            this.renderCharts();
            this.predictBtn.disabled = false;
        } catch (error) {
            this.updateStatus(`Prediction error: ${error.message}`, true);
            console.error('Prediction error:', error);
            this.predictBtn.disabled = false;
        }
    }

    // Render accuracy and timeline charts
    renderCharts() {
        if (!this.evaluationResults) return;

        this.renderAccuracyChart();
        this.renderTimelineChart();
    }

    // Render accuracy bar chart
    renderAccuracyChart() {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        // Destroy previous chart if exists
        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }

        const { stockAccuracies } = this.evaluationResults;
        
        // Sort stocks by accuracy
        const sortedStocks = Object.entries(stockAccuracies)
            .sort(([,a], [,b]) => b - a)
            .map(([symbol, accuracy]) => ({
                symbol,
                accuracy: accuracy * 100 // Convert to percentage
            }));

        const labels = sortedStocks.map(item => item.symbol);
        const data = sortedStocks.map(item => item.accuracy);
        
        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(74, 144, 226, 0.8)');
        gradient.addColorStop(1, 'rgba(74, 144, 226, 0.4)');

        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: data,
                    backgroundColor: gradient,
                    borderColor: 'rgba(74, 144, 226, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#e0e0e0'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#e0e0e0'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#e0e0e0'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Accuracy: ${context.parsed.x.toFixed(2)}%`
                        }
                    }
                }
            }
        });
    }

    // Render prediction timeline chart
    renderTimelineChart() {
        const ctx = document.getElementById('timelineChart').getContext('2d');
        
        // Destroy previous chart if exists
        if (this.timelineChart) {
            this.timelineChart.destroy();
        }

        const { stockAccuracies, stockPredictions } = this.evaluationResults;
        
        // Get top 3 stocks by accuracy
        const topStocks = Object.entries(stockAccuracies)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3)
            .map(([symbol]) => symbol);

        const datasets = topStocks.map((symbol, index) => {
            const predictions = stockPredictions[symbol];
            const colors = ['#4a90e2', '#50e3c2', '#e3507d']; // Different colors for each stock
            
            return {
                label: `${symbol} Predictions`,
                data: predictions.map(p => p.correct),
                borderColor: colors[index],
                backgroundColor: colors[index] + '40',
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6
            };
        });

        const labels = Array.from({length: Math.min(50, stockPredictions[topStocks[0]].length)}, (_, i) => `Pred ${i + 1}`);

        this.timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Prediction Number'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#e0e0e0'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Correct (1) / Incorrect (0)'
                        },
                        min: -0.1,
                        max: 1.1,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#e0e0e0',
                            stepSize: 1,
                            callback: function(value) {
                                return value === 1 ? 'Correct' : value === 0 ? 'Incorrect' : '';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#e0e0e0'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const stock = context.dataset.label.replace(' Predictions', '');
                                const predIndex = context.dataIndex;
                                const prediction = stockPredictions[stock][predIndex];
                                return [
                                    `Stock: ${stock}`,
                                    `Prediction: ${prediction.pred === 1 ? 'UP' : 'DOWN'}`,
                                    `Actual: ${prediction.truth === 1 ? 'UP' : 'DOWN'}`,
                                    `Result: ${prediction.correct === 1 ? 'CORRECT' : 'INCORRECT'}`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }

    // Update status message
    updateStatus(message, isError = false) {
        this.statusElement.textContent = message;
        this.statusElement.style.borderLeftColor = isError ? '#e3507d' : '#4a90e2';
        
        // Auto-clear success messages after 5 seconds
        if (!isError && !message.includes('Please')) {
            setTimeout(() => {
                if (this.statusElement.textContent === message) {
                    this.statusElement.textContent = 'Ready';
                    this.statusElement.style.borderLeftColor = '#4a90e2';
                }
            }, 5000);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
