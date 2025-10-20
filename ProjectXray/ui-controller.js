/**
 * UI Controller for Pneumonia Chest X-ray Classifier
 * Handles user interactions, visualization, and application flow
 * Medical safety warnings and educational context emphasized throughout
 */

class UIController {
    constructor() {
        this.model = new PneumoniaClassifier();
        this.dataLoader = new MedicalImageLoader();
        this.currentMode = 'inference';
        this.isProcessing = false;
        this.initialized = false;
        
        this.initializeApplication();
    }

    /**
     * Initialize the application with TensorFlow.js
     */
    async initializeApplication() {
        try {
            this.showLoading('Initializing AI engine...', true);
            
            // Initialize TensorFlow.js
            const backendInfo = await this.model.initializeTFJS();
            document.getElementById('backendInfo').textContent = backendInfo;
            
            // Check for existing model
            const modelExists = await this.model.modelExists();
            if (modelExists) {
                document.getElementById('backendInfo').textContent += ' | Pre-trained model available';
            }
            
            this.initialized = true;
            this.hideLoading();
            this.initializeEventListeners();
            this.showSafetyWarning();
            
            console.log('Application initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showError(`Failed to initialize AI engine: ${error.message}. Please refresh the page.`);
        }
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
        const normalFiles = document.getElementById('normalFiles');
        const pneumoniaFiles = document.getElementById('pneumoniaFiles');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));
        
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        normalFiles.addEventListener('change', (e) => this.handleTrainingFileSelect(e, 'normal'));
        pneumoniaFiles.addEventListener('change', (e) => this.handleTrainingFileSelect(e, 'pneumonia'));

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
        if (!this.initialized) {
            this.showError('Application not initialized yet. Please wait.');
            return;
        }
        
        this.currentMode = mode;
        
        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        const trainingUpload = document.getElementById('trainingUpload');
        const inferenceUpload = document.getElementById('inferenceUpload');
        const trainingControls = document.getElementById('trainingControls');
        const trainBtn = document.getElementById('trainBtn');
        const processBtn = document.getElementById('processBtn');
        const panelTitle = document.getElementById('panelTitle');
        const visualizationPanel = document.getElementById('visualizationPanel');
        
        if (mode === 'training') {
            trainingUpload.style.display = 'block';
            inferenceUpload.style.display = 'none';
            trainingControls.style.display = 'grid';
            trainBtn.style.display = 'inline-block';
            processBtn.textContent = 'Prepare Training Data';
            panelTitle.textContent = 'Training Data Setup';
            visualizationPanel.style.display = 'block';
            this.showEducationalContext();
        } else {
            trainingUpload.style.display = 'none';
            inferenceUpload.style.display = 'block';
            trainingControls.style.display = 'none';
            trainBtn.style.display = 'none';
            processBtn.textContent = 'Classify Images';
            panelTitle.textContent = 'Image Upload & Processing';
            visualizationPanel.style.display = 'none';
            this.showSafetyWarning();
        }
        
        this.updateResultsArea(`Switched to ${mode} mode. ${this.getModeDescription(mode)}`);
    }

    /**
     * Get mode description for educational context
     */
    getModeDescription(mode) {
        const descriptions = {
            inference: 'Upload chest X-ray images to get AI-powered pneumonia classification predictions.',
            training: 'Educational mode: Upload separate sets of normal and pneumonia chest X-rays to train the AI model.'
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
     * Handle training file selection
     */
    async handleTrainingFileSelect(event, fileType) {
        const files = event.target.files;
        if (files.length > 0) {
            await this.handleTrainingFiles(files, fileType);
        }
    }

    /**
     * Process training files and show statistics
     */
    async handleTrainingFiles(files, fileType) {
        const statsDiv = document.getElementById(`${fileType}Stats`);
        const validFiles = this.dataLoader.validateFiles(files);
        
        if (validFiles.length > 0) {
            statsDiv.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value ${fileType}-stat">${validFiles.length}</div>
                    <div>${fileType.charAt(0).toUpperCase() + fileType.slice(1)} Images</div>
                </div>
            `;
            statsDiv.style.display = 'grid';
            
            // Update training data summary
            this.updateTrainingDataSummary();
        } else {
            statsDiv.style.display = 'none';
        }
        
        this.checkTrainingDataReady();
    }

    /**
     * Update training data summary
     */
    updateTrainingDataSummary() {
        const normalFiles = document.getElementById('normalFiles').files;
        const pneumoniaFiles = document.getElementById('pneumoniaFiles').files;
        
        const normalCount = this.dataLoader.validateFiles(normalFiles).length;
        const pneumoniaCount = this.dataLoader.validateFiles(pneumoniaFiles).length;
        const totalCount = normalCount + pneumoniaCount;
        
        if (totalCount > 0) {
            const balanceRatio = normalCount > 0 ? (normalCount / totalCount * 100).toFixed(1) : '0.0';
            
            document.getElementById('normalCount').textContent = normalCount;
            document.getElementById('pneumoniaCount').textContent = pneumoniaCount;
            document.getElementById('totalCount').textContent = totalCount;
            document.getElementById('balanceRatio').textContent = `${balanceRatio}%`;
            
            document.getElementById('trainingDataSummary').style.display = 'block';
        } else {
            document.getElementById('trainingDataSummary').style.display = 'none';
        }
    }

    /**
     * Check if both training file sets are ready
     */
    checkTrainingDataReady() {
        const normalFiles = document.getElementById('normalFiles').files;
        const pneumoniaFiles = document.getElementById('pneumoniaFiles').files;
        
        const validNormal = this.dataLoader.validateFiles(normalFiles).length;
        const validPneumonia = this.dataLoader.validateFiles(pneumoniaFiles).length;
        
        if (validNormal > 0 && validPneumonia > 0) {
            document.getElementById('processBtn').disabled = false;
            this.updateResultsArea('Training data selected. Click "Prepare Training Data" to continue.');
        } else {
            document.getElementById('processBtn').disabled = true;
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
        if (!this.initialized) {
            this.showError('Application not initialized. Please wait.');
            return;
        }
        
        this.showLoading('Loading and validating medical images...');
        
        try {
            const images = await this.dataLoader.loadTestData(files);
            this.createImagePreviews(images);
            document.getElementById('processBtn').disabled = false;
            this.hideLoading();
            
            this.updateResultsArea(`Successfully loaded ${images.length} medical images. Click "Classify Images" to continue.`);
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
        if (!this.initialized) {
            this.showError('Application not initialized. Please wait.');
            return;
        }
        
        this.isProcessing = true;
        this.showLoading('Processing medical images...');
        document.getElementById('processBtn').disabled = true;
        
        try {
            if (this.currentMode === 'inference') {
                await this.prepareInference();
            } else {
                await this.prepareTraining();
            }
            
        } catch (error) {
            this.showError(`Processing error: ${error.message}`);
        } finally {
            this.hideLoading();
            this.isProcessing = false;
        }
    }

    /**
     * Prepare data for inference mode
     */
    async prepareInference() {
        if (this.dataLoader.loadedImages.length === 0) {
            this.showError('Please upload medical images first.');
            return;
        }
        
        // Preprocess images
        const processedData = this.dataLoader.preprocessImages(this.dataLoader.loadedImages);
        await this.runInference(processedData.tensors);
    }

    /**
     * Prepare data for training mode
     */
    async prepareTraining() {
        const normalFiles = document.getElementById('normalFiles').files;
        const pneumoniaFiles = document.getElementById('pneumoniaFiles').files;
        
        const validNormal = this.dataLoader.validateFiles(normalFiles);
        const validPneumonia = this.dataLoader.validateFiles(pneumoniaFiles);
        
        if (validNormal.length === 0 || validPneumonia.length === 0) {
            this.showError('Please select both normal and pneumonia images for training.');
            return;
        }
        
        if (validNormal.length + validPneumonia.length < 10) {
            this.showError('Please provide at least 10 images total for training (5 normal, 5 pneumonia).');
            return;
        }
        
        this.showLoading('Loading and preparing training data...');
        
        try {
            // Load training data from files
            await this.dataLoader.loadTrainingData(validNormal, validPneumonia);
            
            // Prepare training dataset
            const trainingData = this.dataLoader.prepareTrainingDataset();
            
            // Show training data statistics
            const stats = this.dataLoader.getTrainingStats();
            const distribution = this.dataLoader.getClassDistribution();
            
            this.updateResultsArea(`
                <h3>Training Data Prepared</h3>
                <div class="results-summary">
                    <h4>Dataset Statistics</h4>
                    <div class="folder-stats">
                        <div class="stat-card">
                            <div class="stat-value normal-stat">${stats.normal}</div>
                            <div>Normal Images</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value pneumonia-stat">${stats.pneumonia}</div>
                            <div>Pneumonia Images</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.total}</div>
                            <div>Total Images</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${distribution.normalRatio}%</div>
                            <div>Normal Ratio</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${distribution.pneumoniaRatio}%</div>
                            <div>Pneumonia Ratio</div>
                        </div>
                    </div>
                </div>
                <p>Training data is ready. Click "Start Training" to begin the educational model training process.</p>
            `);
            
            this.trainingData = trainingData;
            document.getElementById('trainBtn').disabled = false;
            
        } catch (error) {
            this.showError(`Training data preparation error: ${error.message}`);
        }
    }

    /**
     * Run inference on processed images
     */
    async runInference(tensors) {
        this.showLoading('Running AI classification...');
        
        try {
            // Try to load existing model first
            let modelLoaded = false;
            try {
                await this.model.loadModel();
                modelLoaded = true;
                this.updateResultsArea('Pre-trained model loaded. Starting classification...');
            } catch (error) {
                console.log('No pre-trained model found, using new model');
                this.model.model = this.model.createModel();
                this.updateResultsArea('New model initialized. Starting classification...');
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
        
        let pneumoniaCount = 0;
        let normalCount = 0;
        let errorCount = 0;
        
        // Create batch results summary
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'batch-results';
        
        const resultsList = document.createElement('div');
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i];
            const imageData = this.dataLoader.loadedImages[i];
            
            if (prediction.error) {
                errorCount++;
                continue;
            }
            
            const resultDiv = document.createElement('div');
            resultDiv.className = `prediction-result ${
                prediction.predictedClass === 'Pneumonia' ? 'pneumonia-positive' : 'normal-negative'
            }`;
            
            // Count results for summary
            if (prediction.predictedClass === 'Pneumonia') pneumoniaCount++;
            else normalCount++;
            
            resultDiv.innerHTML = `
                <h4>${imageData.file.name}</h4>
                <div><strong>Prediction:</strong> ${prediction.predictedClass}</div>
                <div><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(2)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${prediction.confidence * 100}%"></div>
                </div>
                <div><strong>Pneumonia Probability:</strong> ${(prediction.pneumonia * 100).toFixed(2)}%</div>
                <div><strong>Normal Probability:</strong> ${(prediction.normal * 100).toFixed(2)}%</div>
                ${this.getConfidenceWarning(prediction.confidence)}
            `;
            
            resultsList.appendChild(resultDiv);
            
            // Generate and display heatmap for first image
            if (i === 0) {
                const heatmap = await this.model.generateHeatmap(tensors[i]);
                if (heatmap) {
                    this.displayHeatmap(heatmap);
                }
            }
        }
        
        // Show batch summary
        summaryDiv.innerHTML = `
            <div class="results-summary">
                <h4>Batch Summary</h4>
                <div class="folder-stats">
                    <div class="stat-card">
                        <div class="stat-value normal-stat">${normalCount}</div>
                        <div>Normal Predictions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value pneumonia-stat">${pneumoniaCount}</div>
                        <div>Pneumonia Predictions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${predictions.length}</div>
                        <div>Total Images</div>
                    </div>
                    ${errorCount > 0 ? `
                    <div class="stat-card">
                        <div class="stat-value" style="color: var(--warning);">${errorCount}</div>
                        <div>Errors</div>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
        
        resultsArea.appendChild(summaryDiv);
        resultsArea.appendChild(resultsList);
        
        this.showPredictionSummary(pneumoniaCount, normalCount, predictions.length);
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
        
        canvas.width = 150;
        canvas.height = 150;
        
        // Create heatmap visualization
        const imageData = ctx.createImageData(150, 150);
        
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
    showPredictionSummary(pneumoniaCount, normalCount, totalCount) {
        const summary = document.createElement('div');
        summary.className = 'prediction-result';
        summary.style.background = '#fff3cd';
        summary.style.borderLeft = '5px solid var(--warning)';
        
        summary.innerHTML = `
            <h3>📊 Summary Report</h3>
            <div><strong>Total Images Processed:</strong> ${totalCount}</div>
            <div><strong>Pneumonia Predictions:</strong> ${pneumoniaCount}</div>
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
     * Start model training in educational mode
     */
    async startTraining() {
        if (!this.trainingData) {
            this.showError('No training data prepared. Please process images first.');
            return;
        }
        
        this.showLoading('Starting educational model training...');
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('trainingProgress').style.display = 'block';
        
        try {
            // Create model
            this.model.model = this.model.createModel();
            
            // Split data
            const splitData = this.dataLoader.splitData(
                this.trainingData.tensors,
                this.trainingData.labels,
                document.getElementById('testSplit').value / 100
            );
            
            // Create training dataset
            const trainingDataset = this.dataLoader.createTrainingDataset(
                splitData.train.tensors,
                splitData.train.labels
            );
            
            const config = {
                epochs: parseInt(document.getElementById('epochs').value),
                batchSize: parseInt(document.getElementById('batchSize').value),
                validationSplit: document.getElementById('testSplit').value / 100
            };
            
            // Training progress callback
            const onProgress = (progress) => {
                const progressPercent = (progress.epoch / progress.totalEpochs) * 100;
                document.getElementById('progressFill').style.width = `${progressPercent}%`;
                document.getElementById('progressText').textContent = 
                    `Epoch ${progress.epoch}/${progress.totalEpochs} - Loss: ${progress.loss.toFixed(4)} - Accuracy: ${(progress.accuracy * 100).toFixed(2)}%`;
            };
            
            // Train model
            await this.model.trainModel(trainingDataset, splitData.validation, config, onProgress);
            
            // Evaluate model
            if (splitData.validation.tensors.length > 0) {
                const testPredictions = await this.model.predictBatch(splitData.validation.tensors);
                const metrics = this.model.calculateMetrics(testPredictions, splitData.validation.labels);
                this.displayTrainingMetrics(metrics);
            }
            
            // Save model
            await this.model.saveModel();
            
            document.getElementById('trainingProgress').style.display = 'none';
            this.updateResultsArea('Educational training completed! Model is saved and ready for inference.');
            
        } catch (error) {
            document.getElementById('trainingProgress').style.display = 'none';
            this.showError(`Training error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Display training performance metrics
     */
    displayTrainingMetrics(metrics) {
        const metricsGrid = document.getElementById('metricsGrid');
        metricsGrid.innerHTML = `
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value accuracy">${metrics.accuracy}%</div>
                <div>Overall performance</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value precision">${metrics.precision}%</div>
                <div>Pneumonia detection accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Recall</div>
                <div class="metric-value recall">${metrics.recall}%</div>
                <div>Pneumonia detection rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value f1">${metrics.f1Score}%</div>
                <div>Overall balance</div>
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
            <ol>
                <li><strong>Step 1:</strong> Upload normal chest X-ray images (no pneumonia)</li>
                <li><strong>Step 2:</strong> Upload pneumonia chest X-ray images</li>
                <li><strong>Step 3:</strong> Prepare training data and review statistics</li>
                <li><strong>Step 4:</strong> Train the AI model and monitor progress</li>
                <li><strong>Step 5:</strong> Evaluate model performance on validation data</li>
            </ol>
            <div class="medical-disclaimer">
                <strong>Important:</strong> This is for educational purposes only. 
                Real medical AI requires extensive validation, clinical testing, and regulatory approval.
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
    showLoading(message = 'Processing...', forceShow = false) {
        const loading = document.getElementById('loadingIndicator');
        const loadingText = document.getElementById('loadingText');
        
        loadingText.textContent = message;
        
        if (forceShow || this.isProcessing) {
            loading.style.display = 'block';
        }
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
        document.getElementById('trainingProgress').style.display = 'none';
        document.getElementById('processBtn').disabled = true;
        document.getElementById('trainBtn').disabled = true;
        
        // Clear file inputs
        document.getElementById('fileInput').value = '';
        document.getElementById('normalFiles').value = '';
        document.getElementById('pneumoniaFiles').value = '';
        
        // Clear stats
        document.getElementById('normalStats').style.display = 'none';
        document.getElementById('pneumoniaStats').style.display = 'none';
        document.getElementById('trainingDataSummary').style.display = 'none';
        
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
    
    // Initialize the application
    window.app = new UIController();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIController;
}
