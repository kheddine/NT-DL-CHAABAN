/**
 * UI Controller for COVID-19 Chest X-ray Classifier
 * Handles user interactions, visualization, and application flow
 * Medical safety warnings and educational context emphasized throughout
 */

class UIController {
    constructor() {
        this.model = new COVID19Classifier();
        this.dataLoader = new MedicalImageLoader();
        this.currentMode = 'inference';
        this.isProcessing = false;
        
        this.initializeEventListeners();
        this.showSafetyWarning();
    }

    /**
     * Initialize all UI event listeners
     */
    initializeEventListeners() {
        // Mode selection
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchMode(e.target.dataset.mode));
        });

        // File upload handling
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Action buttons
        document.getElementById('processBtn').addEventListener('click', () => this.processImages());
        document.getElementById('trainBtn').addEventListener('click', () => this.startTraining());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetApplication());

        // Keyboard accessibility
        uploadArea.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                fileInput.click();
            }
        });

        // Window unload cleanup
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    /**
     * Switch between inference and training modes
     */
    switchMode(mode) {
        this.currentMode = mode;
        
        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        const trainingControls = document.getElementById('trainingControls');
        const trainBtn = document.getElementById('trainBtn');
        const processBtn = document.getElementById('processBtn');
        
        if (mode === 'training') {
            trainingControls.style.display = 'grid';
            trainBtn.style.display = 'inline-block';
            processBtn.textContent = 'Preprocess Images';
            this.showEducationalContext();
        } else {
            trainingControls.style.display = 'none';
            trainBtn.style.display = 'none';
            processBtn.textContent = 'Classify Images';
            this.showSafetyWarning();
        }
        
        this.updateResultsArea(`Switched to ${mode} mode. ${this.getModeDescription(mode)}`);
    }

    /**
     * Get mode description for educational context
     */
    getModeDescription(mode) {
        const descriptions = {
            inference: 'Upload chest X-ray images to get AI-powered classification predictions.',
            training: 'Educational mode: Train the AI model with sample data to understand how medical AI works.'
        };
        return descriptions[mode] || '';
    }

    /**
     * Handle file selection via input
     */
    async handleFileSelect(event) {
        const files = event.target.files;
        if (files.length > 0) {
            await this.loadAndPreviewImages(files);
        }
    }

    /**
     * Handle drag over event
     */
    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    /**
     * Handle drag leave event
     */
    handleDragLeave(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
    }

    /**
     * Handle file drop event
     */
    async handleFileDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            await this.loadAndPreviewImages(files);
        }
    }

    /**
     * Load images and create previews
     */
    async loadAndPreviewImages(files) {
        this.showLoading('Loading and validating medical images...');
        
        try {
            const images = await this.dataLoader.loadImagesFromFiles(files);
            this.createImagePreviews(images);
            document.getElementById('processBtn').disabled = false;
            this.hideLoading();
            
            this.updateResultsArea(`Successfully loaded ${images.length} medical images. Click "Process Images" to continue.`);
        } catch (error) {
            this.hideLoading();
            this.showError(`Error loading images: ${error.message}`);
        }
    }

    /**
     * Create image preview grid
     */
    createImagePreviews(images) {
        const previewContainer = document.getElementById('imagePreview');
        previewContainer.innerHTML = '';
        
        images.forEach((imageData, index) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.innerHTML = `
                <img src="${imageData.url}" alt="Chest X-ray ${index + 1}" loading="lazy">
                <div style="padding: 5px; font-size: 0.8rem; background: white;">
                    ${imageData.file.name}
                </div>
            `;
            previewContainer.appendChild(previewItem);
        });
    }

    /**
     * Process images based on current mode
     */
    async processImages() {
        if (this.dataLoader.loadedImages.length === 0) {
            this.showError('Please upload medical images first.');
            return;
        }
        
        this.isProcessing = true;
        this.showLoading('Processing medical images...');
        document.getElementById('processBtn').disabled = true;
        
        try {
            // Preprocess images
            const processedData = this.dataLoader.preprocessImages(this.dataLoader.loadedImages);
            
            if (this.currentMode === 'inference') {
                await this.runInference(processedData.tensors);
            } else {
                await this.prepareTraining(processedData);
            }
            
        } catch (error) {
            this.showError(`Processing error: ${error.message}`);
        } finally {
            this.hideLoading();
            this.isProcessing = false;
        }
    }

    /**
     * Run inference on processed images
     */
    async runInference(tensors) {
        this.showLoading('Running AI classification...');
        
        try {
            // Load or create model
            if (!this.model.model) {
                await this.model.createModel();
                this.updateResultsArea('AI model initialized. Starting classification...');
            }
            
            // Run predictions
            const predictions = await this.model.predictBatch(tensors);
            this.displayPredictions(predictions, tensors);
            
        } catch (error) {
            this.showError(`Inference error: ${error.message}`);
        }
    }

    /**
     * Display prediction results with medical safety context
     */
    async displayPredictions(predictions, tensors) {
        const resultsArea = document.getElementById('resultsArea');
        resultsArea.innerHTML = '<h3>Classification Results</h3>';
        
        let covidCount = 0;
        let normalCount = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i];
            if (!prediction) continue;
            
            const resultDiv = document.createElement('div');
            resultDiv.className = `prediction-result ${
                prediction.predictedClass === 'COVID-19' ? 'covid-positive' : 'covid-negative'
            }`;
            
            // Count results for summary
            if (prediction.predictedClass === 'COVID-19') covidCount++;
            else normalCount++;
            
            resultDiv.innerHTML = `
                <h4>Image ${i + 1}</h4>
                <div><strong>Prediction:</strong> ${prediction.predictedClass}</div>
                <div><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(2)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${prediction.confidence * 100}%"></div>
                </div>
                <div><strong>COVID-19 Probability:</strong> ${(prediction.covid * 100).toFixed(2)}%</div>
                <div><strong>Normal Probability:</strong> ${(prediction.normal * 100).toFixed(2)}%</div>
                ${this.getConfidenceWarning(prediction.confidence)}
            `;
            
            resultsArea.appendChild(resultDiv);
            
            // Generate and display heatmap for first image
            if (i === 0) {
                const heatmap = await this.model.generateHeatmap(tensors[i]);
                if (heatmap) {
                    this.displayHeatmap(heatmap);
                }
            }
        }
        
        // Show summary with medical disclaimer
        this.showPredictionSummary(covidCount, normalCount, predictions.length);
        this.showSafetyWarning();
    }

    /**
     * Get confidence level warning for medical context
     */
    getConfidenceWarning(confidence) {
        if (confidence < 0.7) {
            return `<div style="color: var(--warning); font-weight: bold; margin-top: 10px;">
                    ⚠️ Low confidence - Results should not be used for medical decisions
                </div>`;
        } else if (confidence < 0.9) {
            return `<div style="color: var(--warning); margin-top: 10px;">
                    ⚠️ Moderate confidence - Educational purposes only
                </div>`;
        }
        return '';
    }

    /**
     * Display model attention heatmap
     */
    displayHeatmap(heatmapData) {
        const container = document.getElementById('heatmapContainer');
        const canvas = document.getElementById('heatmapCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = 224;
        canvas.height = 224;
        
        // Create heatmap visualization
        const imageData = ctx.createImageData(224, 224);
        
        for (let i = 0; i < heatmapData.length; i++) {
            const intensity = Math.floor(heatmapData[i] * 255);
            const j = i * 4;
            
            // Red color for high attention areas
            imageData.data[j] = intensity;     // Red
            imageData.data[j + 1] = 0;         // Green
            imageData.data[j + 2] = 0;         // Blue
            imageData.data[j + 3] = 150;       // Alpha
        }
        
        ctx.putImageData(imageData, 0, 0);
        container.style.display = 'block';
    }

    /**
     * Show prediction summary with strong medical disclaimers
     */
    showPredictionSummary(covidCount, normalCount, totalCount) {
        const summary = document.createElement('div');
        summary.className = 'prediction-result';
        summary.style.background = '#fff3cd';
        summary.style.borderLeft = '5px solid var(--warning)';
        
        summary.innerHTML = `
            <h3>📊 Summary Report</h3>
            <div><strong>Total Images Processed:</strong> ${totalCount}</div>
            <div><strong>COVID-19 Predictions:</strong> ${covidCount}</div>
            <div><strong>Normal Predictions:</strong> ${normalCount}</div>
            <div style="margin-top: 15px; padding: 10px; background: #f8d7da; border-radius: 5px;">
                <strong>⚠️ MEDICAL DISCLAIMER:</strong><br>
                These AI predictions are for EDUCATIONAL AND RESEARCH purposes only.<br>
                <strong>DO NOT use for clinical diagnosis or medical decision-making.</strong><br>
                Always consult qualified healthcare professionals for medical concerns.
            </div>
        `;
        
        document.getElementById('resultsArea').prepend(summary);
    }

    /**
     * Prepare data for training mode
     */
    async prepareTraining(processedData) {
        this.updateResultsArea('Preparing training data with medical image augmentations...');
        
        // Generate synthetic data for demonstration (in real scenario, use actual labeled data)
        const syntheticData = this.dataLoader.generateSyntheticData(20);
        const augmentedData = this.dataLoader.augmentData(
            syntheticData.tensors, 
            syntheticData.labels, 
            3
        );
        
        // Split data
        const splitData = this.dataLoader.splitData(
            augmentedData.tensors, 
            augmentedData.labels, 
            document.getElementById('testSplit').value / 100
        );
        
        // Create training dataset
        const trainingData = this.dataLoader.createTrainingDataset(
            splitData.train.tensors,
            splitData.train.labels,
            parseInt(document.getElementById('batchSize').value)
        );
        
        this.trainingData = trainingData;
        this.testData = splitData.test;
        
        document.getElementById('trainBtn').disabled = false;
        this.updateResultsArea('Training data prepared. Click "Start Training" to begin educational model training.');
    }

    /**
     * Start model training in educational mode
     */
    async startTraining() {
        if (!this.trainingData) {
            this.showError('No training data prepared. Please process images first.');
            return;
        }
        
        this.showLoading('Starting educational model training...');
        document.getElementById('trainBtn').disabled = true;
        
        try {
            // Create model
            this.model.model = this.model.createModel();
            
            const config = {
                epochs: parseInt(document.getElementById('epochs').value),
                batchSize: parseInt(document.getElementById('batchSize').value),
                validationSplit: document.getElementById('testSplit').value / 100
            };
            
            // Train model
            await this.model.trainModel(this.trainingData, this.testData, config);
            
            // Evaluate model
            await this.evaluateModel();
            
            this.updateResultsArea('Educational training completed! Model is ready for inference.');
            
        } catch (error) {
            this.showError(`Training error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Evaluate model performance
     */
    async evaluateModel() {
        if (!this.testData) return;
        
        const testTensors = this.dataLoader.createBatch(this.testData.tensors);
        const testLabels = this.dataLoader.labelsToOneHot(this.testData.labels);
        
        const evaluation = this.model.model.evaluate(testTensors, testLabels);
        const loss = evaluation[0].dataSync()[0];
        const accuracy = evaluation[1].dataSync()[0];
        
        // Display metrics
        this.displayTrainingMetrics(accuracy);
        
        // Cleanup
        testTensors.dispose();
        testLabels.dispose();
        evaluation.forEach(tensor => tensor.dispose());
    }

    /**
     * Display training performance metrics
     */
    displayTrainingMetrics(accuracy) {
        const metricsGrid = document.getElementById('metricsGrid');
        metricsGrid.innerHTML = `
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value accuracy">${(accuracy * 100).toFixed(2)}%</div>
                <div>Model performance on test data</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Educational Use</div>
                <div class="metric-value">🎓</div>
                <div>Research purposes only</div>
            </div>
        `;
        metricsGrid.style.display = 'grid';
    }

    /**
     * Show educational context for training mode
     */
    showEducationalContext() {
        this.updateResultsArea(`
            <h3>🎓 Educational Training Mode</h3>
            <p>This mode demonstrates how AI models are trained for medical image analysis:</p>
            <ul>
                <li>Learn about convolutional neural networks (CNNs)</li>
                <li>Understand data augmentation techniques</li>
                <li>See training progress in real-time</li>
                <li>Explore model performance metrics</li>
            </ul>
            <div class="medical-disclaimer">
                <strong>Important:</strong> This is a simulation with synthetic data. 
                Real medical AI requires extensive validation and regulatory approval.
            </div>
        `);
    }

    /**
     * Show safety warning
     */
    showSafetyWarning() {
        const existingWarning = document.querySelector('.safety-warning');
        if (existingWarning) return;
        
        const warning = document.createElement('div');
        warning.className = 'medical-disclaimer safety-warning';
        warning.innerHTML = `
            ⚠️ <strong>MEDICAL SAFETY WARNING:</strong> This tool is for EDUCATIONAL AND RESEARCH purposes only. 
            NOT for clinical use. AI predictions may be inaccurate. Always consult healthcare professionals 
            for medical diagnosis and treatment decisions.
        `;
        
        document.getElementById('resultsArea').prepend(warning);
    }

    /**
     * Update results area with formatted content
     */
    updateResultsArea(content) {
        const resultsArea = document.getElementById('resultsArea');
        if (typeof content === 'string') {
            resultsArea.innerHTML = content;
        }
    }

    /**
     * Show loading indicator
     */
    showLoading(message = 'Processing...') {
        const loading = document.getElementById('loadingIndicator');
        loading.style.display = 'block';
        loading.querySelector('p').textContent = message;
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        document.getElementById('loadingIndicator').style.display = 'none';
    }

    /**
     * Show error message
     */
    showError(message) {
        this.updateResultsArea(`
            <div class="prediction-result" style="background: #f8d7da; border-left-color: var(--danger);">
                <h3>❌ Error</h3>
                <p>${message}</p>
                <p><small>Please check the console for technical details.</small></p>
            </div>
        `);
        this.hideLoading();
    }

    /**
     * Reset application to initial state
     */
    resetApplication() {
        // Clear data
        this.dataLoader.clearPreviousData();
        this.model.dispose();
        
        // Reset UI
        document.getElementById('imagePreview').innerHTML = '';
        document.getElementById('resultsArea').innerHTML = `
            <div class="prediction-placeholder">
                <p>Upload chest X-ray images to see predictions and analysis.</p>
            </div>
        `;
        document.getElementById('metricsGrid').style.display = 'none';
        document.getElementById('heatmapContainer').style.display = 'none';
        document.getElementById('processBtn').disabled = true;
        document.getElementById('trainBtn').disabled = true;
        
        // Clear file input
        document.getElementById('fileInput').value = '';
        
        // Show appropriate message based on mode
        if (this.currentMode === 'training') {
            this.showEducationalContext();
        } else {
            this.showSafetyWarning();
        }
        
        console.log('Application reset completed');
    }

    /**
     * Cleanup resources before unload
     */
    cleanup() {
        this.dataLoader.clearPreviousData();
        this.model.dispose();
        
        // Clear any remaining tensors
        const numTensors = tf.memory().numTensors;
        if (numTensors > 0) {
            console.warn(`Cleaning up ${numTensors} remaining tensors`);
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if TensorFlow.js is available
    if (typeof tf === 'undefined') {
        alert('Error: TensorFlow.js not loaded. Please check your internet connection.');
        return;
    }
    
    // Set TensorFlow.js backend and logging
    tf.setBackend('webgl').then(() => {
        console.log('TensorFlow.js backend initialized:', tf.getBackend());
        window.app = new UIController();
    }).catch(error => {
        console.error('TensorFlow.js initialization failed:', error);
        alert('Error initializing AI engine. Please try refreshing the page.');
    });
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIController;
}
