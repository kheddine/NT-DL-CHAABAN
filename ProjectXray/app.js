class PneumoniaApp {
    constructor() {
        this.model = new PneumoniaModel();
        this.currentMode = 'training';
        this.trainingData = {
            normal: [],
            pneumonia: []
        };
        this.testImages = [];
        
        this.initialize();
    }

    async initialize() {
        try {
            await this.model.initialize();
            this.setupEventListeners();
            this.updateUI();
            this.showResults('Application ready. Upload training data to begin.');
        } catch (error) {
            this.showResults(`Initialization failed: ${error.message}`);
        }
    }

    setupEventListeners() {
        // Mode switching
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchMode(e.target.dataset.mode);
            });
        });

        // Training file inputs
        document.getElementById('normalFiles').addEventListener('change', (e) => {
            this.handleTrainingFiles(e.target.files, 'normal');
        });
        document.getElementById('pneumoniaFiles').addEventListener('change', (e) => {
            this.handleTrainingFiles(e.target.files, 'pneumonia');
        });

        // Testing file input
        const uploadArea = document.getElementById('uploadArea');
        const testFilesInput = document.getElementById('testFiles');
        
        uploadArea.addEventListener('click', () => testFilesInput.click());
        testFilesInput.addEventListener('change', (e) => {
            this.handleTestFiles(e.target.files);
        });

        // Drag and drop for testing
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#e3f2fd';
            uploadArea.style.borderColor = '#3498db';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
            uploadArea.style.borderColor = '#bdc3c7';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
            uploadArea.style.borderColor = '#bdc3c7';
            this.handleTestFiles(e.dataTransfer.files);
        });

        // Action buttons
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.runPredictions();
        });
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetTraining();
        });
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearTesting();
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        document.getElementById('trainingSection').style.display = 
            mode === 'training' ? 'block' : 'none';
        document.getElementById('testingSection').style.display = 
            mode === 'testing' ? 'block' : 'none';
        
        document.getElementById('panelTitle').textContent = 
            mode === 'training' ? 'Training Data' : 'Test Images';
        document.getElementById('resultsTitle').textContent = 
            mode === 'training' ? 'Training Results' : 'Prediction Results';
        
        this.updateUI();
    }

    async handleTrainingFiles(files, type) {
        const validFiles = this.validateFiles(files);
        if (validFiles.length === 0) return;

        this.showLoading(`Loading ${type} images...`);
        
        try {
            this.trainingData[type] = await this.loadImages(validFiles);
            this.updateTrainingStats(type, this.trainingData[type].length);
            this.checkTrainingReady();
        } catch (error) {
            this.showResults(`Error loading ${type} images: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async handleTestFiles(files) {
        const validFiles = this.validateFiles(files);
        if (validFiles.length === 0) return;

        this.showLoading('Loading test images...');
        
        try {
            this.testImages = await this.loadImages(validFiles);
            this.showImagePreviews(this.testImages);
            document.getElementById('predictBtn').disabled = false;
            this.showResults(`Loaded ${this.testImages.length} test images. Ready for prediction.`);
        } catch (error) {
            this.showResults(`Error loading test images: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    validateFiles(files) {
        return Array.from(files).filter(file => {
            const isImage = file.type.startsWith('image/');
            const isSizeValid = file.size <= 10 * 1024 * 1024; // 10MB
            
            if (!isImage) {
                console.warn(`Skipping non-image file: ${file.name}`);
                return false;
            }
            if (!isSizeValid) {
                console.warn(`File too large: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`);
                return false;
            }
            
            return true;
        });
    }

    async loadImages(files) {
        const images = [];
        for (let file of files) {
            try {
                const imageData = await this.loadImage(file);
                images.push(imageData);
            } catch (error) {
                console.error(`Failed to load image ${file.name}:`, error);
            }
        }
        return images;
    }

    loadImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            const img = new Image();
            
            reader.onload = (e) => {
                img.onload = () => resolve({
                    file: file,
                    element: img,
                    url: e.target.result
                });
                img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
                img.src = e.target.result;
            };
            
            reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
            reader.readAsDataURL(file);
        });
    }

    updateTrainingStats(type, count) {
        const statsDiv = document.getElementById(`${type}Stats`);
        statsDiv.innerHTML = `
            <div class="stat-card">
                <div class="stat-value ${type}-stat">${count}</div>
                <div>${type.charAt(0).toUpperCase() + type.slice(1)} Images</div>
            </div>
        `;
        statsDiv.style.display = 'grid';
    }

    checkTrainingReady() {
        const normalCount = this.trainingData.normal.length;
        const pneumoniaCount = this.trainingData.pneumonia.length;
        const isReady = normalCount > 0 && pneumoniaCount > 0;
        
        document.getElementById('trainBtn').disabled = !isReady;
        
        if (isReady) {
            const total = normalCount + pneumoniaCount;
            this.showResults(`Ready to train! ${total} total images (${normalCount} normal, ${pneumoniaCount} pneumonia)`);
        }
    }

    async startTraining() {
        const normalCount = this.trainingData.normal.length;
        const pneumoniaCount = this.trainingData.pneumonia.length;
        
        if (normalCount === 0 || pneumoniaCount === 0) {
            this.showResults('Please upload both normal and pneumonia images for training.');
            return;
        }

        this.showLoading('Preparing training data...');
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('trainingProgress').style.display = 'block';

        try {
            const trainingData = this.prepareTrainingData();
            const epochs = parseInt(document.getElementById('epochs').value);
            const batchSize = parseInt(document.getElementById('batchSize').value);

            await this.model.train(trainingData, epochs, batchSize, (progress) => {
                const percent = (progress.epoch / progress.totalEpochs) * 100;
                document.getElementById('progressFill').style.width = `${percent}%`;
                
                let progressText = `Epoch ${progress.epoch}/${progress.totalEpochs}`;
                progressText += ` • Accuracy: ${(progress.accuracy * 100).toFixed(1)}%`;
                progressText += ` • Loss: ${progress.loss.toFixed(4)}`;
                
                if (progress.valAccuracy) {
                    progressText += ` • Val Accuracy: ${(progress.valAccuracy * 100).toFixed(1)}%`;
                }
                
                document.getElementById('progressText').textContent = progressText;
            });

            this.showResults('Training completed successfully! Model is ready for testing.');
            
        } catch (error) {
            this.showResults(`Training failed: ${error.message}`);
        } finally {
            this.hideLoading();
            document.getElementById('trainingProgress').style.display = 'none';
            document.getElementById('trainBtn').disabled = false;
        }
    }

    prepareTrainingData() {
        const allTensors = [];
        const allLabels = [];
        
        // Process normal images
        this.trainingData.normal.forEach(img => {
            const tensor = this.preprocessImage(img.element);
            allTensors.push(tensor);
            allLabels.push(0); // Normal = 0
        });
        
        // Process pneumonia images
        this.trainingData.pneumonia.forEach(img => {
            const tensor = this.preprocessImage(img.element);
            allTensors.push(tensor);
            allLabels.push(1); // Pneumonia = 1
        });
        
        const xs = tf.stack(allTensors);
        const ys = tf.oneHot(allLabels, 2);
        
        return { xs, ys };
    }

    preprocessImage(imgElement) {
        return tf.tidy(() => {
            let tensor = tf.browser.fromPixels(imgElement);
            
            // Handle different channel formats
            if (tensor.shape[2] === 1) {
                // Grayscale to RGB
                tensor = tensor.concat([tensor, tensor], 2);
            } else if (tensor.shape[2] === 4) {
                // Remove alpha channel
                tensor = tensor.slice([0, 0, 0], [tensor.shape[0], tensor.shape[1], 3]);
            }
            
            // Resize and normalize
            tensor = tf.image.resizeBilinear(tensor, [150, 150]);
            tensor = tensor.div(255.0);
            
            return tensor;
        });
    }

    async runPredictions() {
        if (this.testImages.length === 0) {
            this.showResults('Please upload test images first.');
            return;
        }

        this.showLoading('Running predictions...');
        document.getElementById('predictBtn').disabled = true;

        try {
            const modelLoaded = await this.model.loadModel();
            if (!modelLoaded) {
                throw new Error('No trained model found. Please train a model first.');
            }

            const testTensors = this.testImages.map(img => this.preprocessImage(img.element));
            const predictions = await this.model.predictBatch(testTensors);
            
            this.displayPredictions(predictions);
            
        } catch (error) {
            this.showResults(`Prediction error: ${error.message}`);
        } finally {
            this.hideLoading();
            document.getElementById('predictBtn').disabled = false;
            
            // Clean up tensors
            tf.engine().startScope();
            tf.engine().endScope();
        }
    }

    displayPredictions(predictions) {
        let html = '<h3>Prediction Results</h3>';
        let normalCount = 0;
        let pneumoniaCount = 0;
        let errorCount = 0;

        // Summary stats
        predictions.forEach((pred, index) => {
            if (!pred) {
                errorCount++;
                return;
            }
            
            if (pred.predictedClass === 'Normal') normalCount++;
            else pneumoniaCount++;
        });

        html += `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value normal-stat">${normalCount}</div>
                    <div>Normal</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value pneumonia-stat">${pneumoniaCount}</div>
                    <div>Pneumonia</div>
                </div>
                ${errorCount > 0 ? `
                <div class="stat-card">
                    <div class="stat-value" style="color: #f39c12;">${errorCount}</div>
                    <div>Errors</div>
                </div>
                ` : ''}
            </div>
        `;

        // Individual predictions
        predictions.forEach((pred, index) => {
            if (!pred) return;
            
            const image = this.testImages[index];
            const isPneumonia = pred.predictedClass === 'Pneumonia';
            const confidencePercent = (pred.confidence * 100).toFixed(1);
            
            html += `
                <div class="prediction-result ${isPneumonia ? 'pneumonia-positive' : 'normal-negative'}">
                    <h4>${image.file.name}</h4>
                    <div><strong>Prediction:</strong> ${pred.predictedClass}</div>
                    <div><strong>Confidence:</strong> ${confidencePercent}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                    <div style="font-size: 0.9em; color: #666;">
                        Normal: ${(pred.normal * 100).toFixed(1)}% | 
                        Pneumonia: ${(pred.pneumonia * 100).toFixed(1)}%
                    </div>
                </div>
            `;
        });

        this.showResults(html);
    }

    showImagePreviews(images) {
        const container = document.getElementById('imagePreview');
        container.innerHTML = '';
        
        images.forEach(img => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `<img src="${img.url}" alt="${img.file.name}">`;
            container.appendChild(div);
        });
    }

    showLoading(message) {
        document.getElementById('loadingText').textContent = message;
        document.getElementById('loadingIndicator').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loadingIndicator').style.display = 'none';
    }

    showResults(content) {
        document.getElementById('resultsArea').innerHTML = content;
    }

    updateUI() {
        if (this.currentMode === 'testing') {
            document.getElementById('predictBtn').disabled = this.testImages.length === 0;
        }
    }

    clearTesting() {
        this.testImages = [];
        document.getElementById('imagePreview').innerHTML = '';
        document.getElementById('testFiles').value = '';
        document.getElementById('predictBtn').disabled = true;
        this.showResults('Upload test images to run predictions.');
    }

    resetTraining() {
        this.trainingData = { normal: [], pneumonia: [] };
        this.model.dispose();
        
        // Reset UI
        document.getElementById('normalFiles').value = '';
        document.getElementById('pneumoniaFiles').value = '';
        document.getElementById('normalStats').style.display = 'none';
        document.getElementById('pneumoniaStats').style.display = 'none';
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('trainingProgress').style.display = 'none';
        
        this.showResults('Training data cleared. Upload new data to begin training.');
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PneumoniaApp();
});
