/**
 * UI Controller for COVID-19 Chest X-ray Classifier
 * Handles user interactions, visualizations, and medical safety features
 * FOR EDUCATIONAL PURPOSES ONLY
 */

class UIController {
    constructor() {
        this.currentMode = 'inference';
        this.uploadedImages = [];
        this.isModelReady = false;
        this.trainingCharts = null;
        this.initializeEventListeners();
        this.initializeModel();
    }

    /**
     * Initialize TensorFlow.js model with medical safety checks
     */
    async initializeModel() {
        try {
            this.showLoading('inference', 'Initializing medical AI model...');
            
            // Check TensorFlow.js backend compatibility
            await tf.ready();
            console.log('TensorFlow.js backend:', tf.getBackend());
            
            // Try to load existing model, otherwise create new one
            try {
                await covidModel.loadModel();
                this.isModelReady = true;
                this.updateModelStatus('Model loaded successfully');
                console.log('Pretrained model loaded');
            } catch (error) {
                // Create new model if none exists
                covidModel.createCNNModel();
                this.isModelReady = true;
                this.updateModelStatus('New model created - ready for training');
                console.log('New model created');
            }
            
            this.hideLoading('inference');
        } catch (error) {
            console.error('Model initialization failed:', error);
            this.showError('Failed to initialize model: ' + error.message);
            this.hideLoading('inference');
        }
    }

    /**
     * Initialize all UI event listeners
     */
    initializeEventListeners() {
        // Mode switching
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchMode(e.target.dataset.mode);
            });
        });

        // File upload handlers
        this.setupFileUpload('inference');
        this.setupFileUpload('training');

        // Control buttons
        document.getElementById('classify-btn').addEventListener('click', () => this.classifyImages());
        document.getElementById('train-btn').addEventListener('click', () => this.startTraining());
        document.getElementById('load-pretrained-btn').addEventListener('click', () => this.loadPretrainedModel());
        
        // Clear buttons
        document.getElementById('clear-inference-btn').addEventListener('click', () => this.clearInference());
        document.getElementById('clear-training-btn').addEventListener('click', () => this.clearTraining());

        // Drag and drop
        this.setupDragAndDrop('inference');
        this.setupDragAndDrop('training');

        // Memory management
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    /**
     * Switch between inference and training modes
     */
    switchMode(mode) {
        this.currentMode = mode;
        
        // Update active buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        // Show/hide mode content
        document.querySelectorAll('.mode-content').forEach(content => {
            content.classList.toggle('active', content.id === `${mode}-mode`);
        });
        
        // Update medical disclaimer based on mode
        this.updateMedicalDisclaimer(mode);
    }

    /**
     * Update medical disclaimer based on current mode
     */
    updateMedicalDisclaimer(mode) {
        const disclaimer = document.querySelector('.medical-disclaimer p');
        if (mode === 'training') {
            disclaimer.innerHTML = `<strong>FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY</strong><br>
            This training mode is for educational demonstration only. Do not use for clinical diagnosis. 
            Model performance in this educational environment does not reflect real-world clinical accuracy.`;
        } else {
            disclaimer.innerHTML = `<strong>FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY</strong><br>
            This application is NOT for clinical diagnosis. Do not use for real medical decisions. 
            Always consult qualified healthcare professionals for medical diagnosis.`;
        }
    }

    /**
     * Setup file upload for specific mode
     */
    setupFileUpload(mode) {
        const fileInput = document.getElementById(`${mode}-file-input`);
        const uploadArea = document.getElementById(`${mode}-upload`);

        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files, mode);
        });
    }

    /**
     * Setup drag and drop functionality
     */
    setupDragAndDrop(mode) {
        const uploadArea = document.getElementById(`${mode}-upload`);

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelection(e.dataTransfer.files, mode);
        });
    }

    /**
     * Handle file selection with medical image validation
     */
    async handleFileSelection(files, mode) {
        if (files.length === 0) return;

        this.showLoading(mode, 'Processing medical images...');

        try {
            const result = await medicalDataLoader.loadMultipleImages(Array.from(files));
            
            if (mode === 'inference') {
                this.uploadedImages = result.results;
                this.displayImagePreviews(result.results, mode);
            } else {
                this.displayImagePreviews(result.results, mode);
            }

            if (result.errors.length > 0) {
                this.showError(`Failed to process ${result.errors.length} files. Check console for details.`);
            }

            this.updateUploadStats(result.results.length, mode);
            this.hideLoading(mode);

        } catch (error) {
            this.showError('Error processing images: ' + error.message);
            this.hideLoading(mode);
        }
    }

    /**
     * Display image previews with medical context
     */
    displayImagePreviews(images, mode) {
        const previewContainer = document.getElementById(`${mode}-preview`);
        previewContainer.innerHTML = '';

        images.forEach((imgData, index) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            
            const img = document.createElement('img');
            img.src = URL.createObjectURL(imgData.originalFile);
            img.alt = `Chest X-ray ${index + 1}`;
            
            const label = document.createElement('div');
            label.className = 'preview-label';
            label.textContent = `Image ${index + 1}`;
            label.style.cssText = 'padding: 5px; background: rgba(0,0,0,0.7); color: white; font-size: 12px;';
            
            previewItem.appendChild(img);
            previewItem.appendChild(label);
            previewContainer.appendChild(previewItem);
        });
    }

    /**
     * Update upload statistics
     */
    updateUploadStats(count, mode) {
        const uploadArea = document.getElementById(`${mode}-upload`);
        const existingStats = uploadArea.querySelector('.upload-stats');
        
        if (existingStats) {
            existingStats.remove();
        }

        const stats = document.createElement('div');
        stats.className = 'upload-stats';
        stats.style.cssText = 'margin-top: 10px; padding: 5px; background: rgba(52, 152, 219, 0.1); border-radius: 4px;';
        stats.textContent = `${count} image(s) loaded successfully`;
        
        uploadArea.appendChild(stats);
    }

    /**
     * Classify uploaded images with medical safety warnings
     */
    async classifyImages() {
        if (!this.isModelReady) {
            this.showError('Model not ready. Please wait for initialization or train a model first.');
            return;
        }

        if (this.uploadedImages.length === 0) {
            this.showError('Please upload chest X-ray images first.');
            return;
        }

        this.showLoading('inference', 'Analyzing chest X-ray images...');

        try {
            const results = await covidModel.predictBatch(
                this.uploadedImages.map(img => img.tensor)
            );

            this.displayClassificationResults(results);
            this.hideLoading('inference');

        } catch (error) {
            this.showError('Classification failed: ' + error.message);
            this.hideLoading('inference');
        }
    }

    /**
     * Display classification results with medical disclaimers
     */
    displayClassificationResults(results) {
        const resultsContainer = document.getElementById('inference-results');
        resultsContainer.innerHTML = '<h3>Classification Results</h3>';

        results.forEach((result, index) => {
            const resultElement = document.createElement('div');
            resultElement.className = `prediction-result ${
                result.className === 'COVID-19' ? 'prediction-covid' : 'prediction-normal'
            }`;

            const confidencePercent = (result.confidence * 100).toFixed(1);
            const confidenceColor = this.getConfidenceColor(result.confidenceLevel);

            resultElement.innerHTML = `
                <div style="display: flex; justify-content: between; align-items: center;">
                    <strong>Image ${index + 1}: ${result.className}</strong>
                    <span style="background: ${confidenceColor}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                        ${result.confidenceLevel.toUpperCase()} CONFIDENCE
                    </span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%; background: ${confidenceColor};"></div>
                </div>
                <div style="font-size: 14px;">
                    Confidence: ${confidencePercent}%<br>
                    COVID-19 Probability: ${(result.probabilities['COVID-19'] * 100).toFixed(1)}%<br>
                    Normal Probability: ${(result.probabilities['Normal'] * 100).toFixed(1)}%
                </div>
                ${this.getMedicalWarning(result)}
            `;

            resultsContainer.appendChild(resultElement);
        });

        // Add comprehensive medical disclaimer
        const disclaimer = document.createElement('div');
        disclaimer.className = 'medical-disclaimer';
        disclaimer.style.marginTop = '20px';
        disclaimer.innerHTML = `
            <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong>
            <p>These results are for educational purposes only. Do not make medical decisions based on these predictions.</p>
            <p>False positives and false negatives are possible. Always consult qualified healthcare professionals.</p>
        `;
        resultsContainer.appendChild(disclaimer);
    }

    /**
     * Get appropriate medical warning based on prediction
     */
    getMedicalWarning(result) {
        if (result.className === 'COVID-19' && result.confidenceLevel === 'low') {
            return `<div style="color: #e74c3c; font-weight: bold; margin-top: 8px;">
                ⚠️ Low confidence COVID-19 prediction - Requires clinical verification
            </div>`;
        }
        
        if (result.confidenceLevel === 'very-low') {
            return `<div style="color: #e74c3c; font-weight: bold; margin-top: 8px;">
                ⚠️ Very low confidence - Results may not be reliable
            </div>`;
        }
        
        return '';
    }

    /**
     * Get color for confidence level
     */
    getConfidenceColor(level) {
        const colors = {
            'high': '#27ae60',
            'medium': '#f39c12',
            'low': '#e67e22',
            'very-low': '#e74c3c'
        };
        return colors[level] || '#95a5a6';
    }

    /**
     * Start model training with educational context
     */
    async startTraining() {
        if (!medicalDataLoader.trainingData) {
            this.showError('Please upload training data first.');
            return;
        }

        this.showLoading('training', 'Training medical AI model...');

        try {
            const epochs = parseInt(document.getElementById('epochs').value) || 10;
            const batchSize = parseInt(document.getElementById('batch-size').value) || 8;
            const learningRate = parseFloat(document.getElementById('learning-rate').value) || 0.001;

            // Initialize training charts
            this.initializeTrainingCharts();

            const history = await covidModel.trainModel(
                medicalDataLoader.trainingData,
                medicalDataLoader.validationData,
                {
                    epochs,
                    batchSize,
                    learningRate,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            this.updateTrainingProgress(epoch, logs, epochs);
                            this.updateTrainingCharts(logs);
                        },
                        onTrainEnd: () => {
                            this.onTrainingComplete();
                        }
                    }
                }
            );

            await covidModel.saveModel();
            this.isModelReady = true;

        } catch (error) {
            this.showError('Training failed: ' + error.message);
            this.hideLoading('training');
        }
    }

    /**
     * Initialize training visualization charts
     */
    initializeTrainingCharts() {
        const chartsContainer = document.getElementById('training-charts');
        chartsContainer.innerHTML = '<div id="training-vis"></div>';

        this.trainingCharts = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
    }

    /**
     * Update training progress display
     */
    updateTrainingProgress(epoch, logs, totalEpochs) {
        const progress = ((epoch + 1) / totalEpochs) * 100;
        
        const progressHTML = `
            <div class="training-progress">
                <h4>Training Progress: Epoch ${epoch + 1}/${totalEpochs}</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progress}%"></div>
                </div>
                <div style="font-size: 14px;">
                    Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(1)}%<br>
                    Validation Loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'} | 
                    Validation Accuracy: ${logs.val_acc ? (logs.val_acc * 100).toFixed(1) + '%' : 'N/A'}
                </div>
            </div>
        `;

        const progressElement = document.getElementById('training-charts');
        progressElement.innerHTML = progressHTML;
    }

    /**
     * Update training charts with new data
     */
    updateTrainingCharts(logs) {
        // Simplified chart update - in production, use tfjs-vis for detailed charts
        this.trainingCharts.loss.push(logs.loss);
        this.trainingCharts.accuracy.push(logs.acc);
        this.trainingCharts.valLoss.push(logs.val_loss);
        this.trainingCharts.valAccuracy.push(logs.val_acc);
    }

    /**
     * Handle training completion
     */
    onTrainingComplete() {
        this.hideLoading('training');
        this.showSuccess('Model training completed successfully!');
        this.displayTrainingMetrics();
    }

    /**
     * Display training metrics and performance
     */
    displayTrainingMetrics() {
        const metricsContainer = document.getElementById('training-metrics');
        
        const finalLoss = this.trainingCharts.loss[this.trainingCharts.loss.length - 1];
        const finalAccuracy = this.trainingCharts.accuracy[this.trainingCharts.accuracy.length - 1];
        const finalValLoss = this.trainingCharts.valLoss[this.trainingCharts.valLoss.length - 1];
        const finalValAccuracy = this.trainingCharts.valAccuracy[this.trainingCharts.valAccuracy.length - 1];

        metricsContainer.innerHTML = `
            <div class="metric-card">
                <div>Final Loss</div>
                <div class="metric-value">${finalLoss.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div>Final Accuracy</div>
                <div class="metric-value">${(finalAccuracy * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div>Validation Loss</div>
                <div class="metric-value">${finalValLoss.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div>Validation Accuracy</div>
                <div class="metric-value">${(finalValAccuracy * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    /**
     * Load pretrained model
     */
    async loadPretrainedModel() {
        this.showLoading('training', 'Loading pretrained model...');
        
        try {
            await covidModel.loadModel();
            this.isModelReady = true;
            this.showSuccess('Pretrained model loaded successfully!');
            this.hideLoading('training');
        } catch (error) {
            this.showError('Failed to load pretrained model: ' + error.message);
            this.hideLoading('training');
        }
    }

    /**
     * Clear inference data
     */
    clearInference() {
        this.uploadedImages = [];
        document.getElementById('inference-preview').innerHTML = '';
        document.getElementById('inference-results').innerHTML = '';
        medicalDataLoader.dispose();
    }

    /**
     * Clear training data
     */
    clearTraining() {
        document.getElementById('training-preview').innerHTML = '';
        document.getElementById('training-metrics').innerHTML = '';
        document.getElementById('training-charts').innerHTML = '';
        medicalDataLoader.dispose();
    }

    /**
     * Show loading indicator
     */
    showLoading(mode, message = 'Loading...') {
        const loadingElement = document.getElementById(`${mode}-loading`);
        loadingElement.style.display = 'block';
        if (message) {
            loadingElement.querySelector('p').textContent = message;
        }
    }

    /**
     * Hide loading indicator
     */
    hideLoading(mode) {
        const loadingElement = document.getElementById(`${mode}-loading`);
        loadingElement.style.display = 'none';
    }

    /**
     * Show error message with medical context
     */
    showError(message) {
        // Create or update error display
        let errorElement = document.getElementById('error-message');
        if (!errorElement) {
            errorElement = document.createElement('div');
            errorElement.id = 'error-message';
            errorElement.style.cssText = `
                position: fixed; top: 20px; right: 20px; 
                background: #e74c3c; color: white; padding: 15px; 
                border-radius: 5px; z-index: 1000; max-width: 400px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            `;
            document.body.appendChild(errorElement);
        }

        errorElement.innerHTML = `
            <strong>⚠️ Error</strong><br>${message}
            <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: white; cursor: pointer;">×</button>
        `;

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorElement.parentElement) {
                errorElement.remove();
            }
        }, 5000);
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        const successElement = document.createElement('div');
        successElement.style.cssText = `
            position: fixed; top: 20px; right: 20px; 
            background: #27ae60; color: white; padding: 15px; 
            border-radius: 5px; z-index: 1000; max-width: 400px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        `;
        successElement.innerHTML = `
            <strong>✅ Success</strong><br>${message}
            <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: white; cursor: pointer;">×</button>
        `;
        
        document.body.appendChild(successElement);
        
        setTimeout(() => {
            if (successElement.parentElement) {
                successElement.remove();
            }
        }, 3000);
    }

    /**
     * Update model status display
     */
    updateModelStatus(message) {
        // Could be implemented to show model status in UI
        console.log('Model status:', message);
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        medicalDataLoader.dispose();
        covidModel.dispose();
        
        // Revoke object URLs
        this.uploadedImages.forEach(img => {
            if (img.originalImage && img.originalImage.src) {
                URL.revokeObjectURL(img.originalImage.src);
            }
        });
    }
}

// Initialize UI Controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const uiController = new UIController();
    
    // Make available globally for debugging
    window.uiController = uiController;
    window.medicalDataLoader = medicalDataLoader;
    window.covidModel = covidModel;
});
