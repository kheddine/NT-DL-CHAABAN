// app.js
class RockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.animationId = null;
        this.lastInferenceTime = 0;
        this.inferenceInterval = 300; // ms between inferences
        
        this.classNames = ['Rock', 'Paper', 'Scissors'];
        this.initializeElements();
        this.initializeEventListeners();
        this.loadModel();
    }

    initializeElements() {
        // Get DOM elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultDiv = document.getElementById('result');
        this.confidenceDiv = document.getElementById('confidence');
        this.statusDiv = document.getElementById('status');
        this.startBtn = document.getElementById('startBtn');
        this.classifyBtn = document.getElementById('classifyBtn');
        this.stopBtn = document.getElementById('stopBtn');
    }

    initializeEventListeners() {
        // Button event listeners
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.classifyBtn.addEventListener('click', () => this.startClassification());
        this.stopBtn.addEventListener('click', () => this.stopClassification());

        // Error handling for video
        this.video.addEventListener('error', (e) => {
            this.updateStatus('Video error: ' + e.message);
        });

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    async loadModel() {
        try {
            this.updateStatus('Loading TensorFlow.js model...');
            this.updateResult('Initializing...');

            // Wait for TensorFlow.js to be ready
            await tf.ready();
            console.log('TensorFlow.js backend:', tf.getBackend());

            // Load model - update this path to your actual model
            const modelPath = './model/model.json';
            this.model = await tf.loadLayersModel(modelPath);
            
            // Warm up the model
            await this.warmUpModel();
            
            this.isModelLoaded = true;
            this.updateStatus('Model loaded successfully! Click "Start Camera" to begin.');
            this.updateResult('Ready to classify');
            
            console.log('Model loaded and warmed up');

        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('Error loading model: ' + error.message);
            this.updateResult('Model loading failed');
        }
    }

    async warmUpModel() {
        // Warm up the model with a dummy inference
        const warmupTensor = tf.zeros([1, 224, 224, 3]);
        const warmupResult = this.model.predict(warmupTensor);
        await warmupResult.data();
        warmupTensor.dispose();
        warmupResult.dispose();
    }

    async startCamera() {
        try {
            this.updateStatus('Requesting camera access...');
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = stream;

            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play().then(resolve);
                };
            });

            this.startBtn.disabled = true;
            this.classifyBtn.disabled = false;
            this.updateStatus('Camera started. Click "Start Classification" to begin.');

        } catch (error) {
            console.error('Error accessing camera:', error);
            let errorMessage = 'Error accessing camera: ';
            
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Camera permission denied. Please allow camera access.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found. Please connect a camera.';
            } else {
                errorMessage += error.message;
            }
            
            this.updateStatus(errorMessage);
        }
    }

    async startClassification() {
        if (!this.isModelLoaded) {
            this.updateStatus('Model not loaded yet. Please wait.');
            return;
        }

        if (!this.video.srcObject) {
            this.updateStatus('Please start camera first.');
            return;
        }

        this.isClassifying = true;
        this.classifyBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.lastInferenceTime = 0;

        this.updateStatus('Classification started...');
        this.classifyFrame();
    }

    stopClassification() {
        this.isClassifying = false;
        this.classifyBtn.disabled = false;
        this.stopBtn.disabled = true;

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        this.updateStatus('Classification stopped.');
        this.confidenceDiv.innerHTML = '';
    }

    async classifyFrame() {
        if (!this.isClassifying) return;

        const now = Date.now();
        
        // Throttle inference to maintain performance
        if (now - this.lastInferenceTime >= this.inferenceInterval) {
            await this.runInference();
            this.lastInferenceTime = now;
        }

        this.animationId = requestAnimationFrame(() => this.classifyFrame());
    }

    async runInference() {
        try {
            // Capture and preprocess frame
            const tensor = this.captureAndPreprocessFrame();
            
            // Run prediction
            const startTime = performance.now();
            const predictions = await this.model.predict(tensor).data();
            const inferenceTime = performance.now() - startTime;
            
            // Process results
            this.processPredictions(predictions, inferenceTime);
            
            // Clean up memory
            tensor.dispose();

        } catch (error) {
            console.error('Error during inference:', error);
            this.updateStatus('Inference error: ' + error.message);
        }
    }

    captureAndPreprocessFrame() {
        return tf.tidy(() => {
            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, 224, 224);
            
            // Convert to tensor and preprocess for typical image models
            return tf.browser.fromPixels(this.canvas)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims(0);
        });
    }

    processPredictions(predictions, inferenceTime) {
        // Find the highest confidence prediction
        let maxConfidence = 0;
        let predictedClass = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] > maxConfidence) {
                maxConfidence = predictions[i];
                predictedClass = i;
            }
        }

        const className = this.classNames[predictedClass];
        this.updateResults(className, predictions, inferenceTime);
    }

    updateResults(className, predictions, inferenceTime) {
        // Update main result with animation
        const confidence = (predictions[this.classNames.indexOf(className)] * 100).toFixed(1);
        this.resultDiv.innerHTML = `
            <div style="font-size: 1.5em; margin-bottom: 10px;">${className}</div>
            <div style="font-size: 1.1em; color: #666;">Confidence: ${confidence}%</div>
            <div style="font-size: 0.8em; color: #999;">Inference: ${inferenceTime.toFixed(1)}ms</div>
        `;
        
        this.resultDiv.classList.add('pulse');
        setTimeout(() => this.resultDiv.classList.remove('pulse'), 500);

        // Update confidence bars
        this.updateConfidenceBars(predictions);
    }

    updateConfidenceBars(predictions) {
        let confidenceHTML = '<div style="margin-bottom: 15px; font-weight: 600;">Confidence Levels:</div>';
        
        this.classNames.forEach((name, index) => {
            const confidence = predictions[index] * 100;
            const barWidth = Math.max(confidence, 2); // Minimum width for visibility
            
            confidenceHTML += `
                <div class="confidence-bar">
                    <div class="confidence-label">
                        <span>${name}</span>
                        <span>${confidence.toFixed(1)}%</span>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: ${barWidth}%;"></div>
                    </div>
                </div>
            `;
        });
        
        this.confidenceDiv.innerHTML = confidenceHTML;
    }

    updateStatus(message) {
        if (this.statusDiv) {
            this.statusDiv.textContent = message;
        }
    }

    updateResult(message) {
        if (this.resultDiv) {
            this.resultDiv.textContent = message;
        }
    }

    cleanup() {
        // Stop classification
        this.stopClassification();
        
        // Stop camera stream
        if (this.video.srcObject) {
            const tracks = this.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }
        
        // Clean up TensorFlow.js memory
        if (this.model) {
            this.model.dispose();
        }
        
        tf.disposeVariables();
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RockPaperScissorsClassifier();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, clean up resources
        const classifier = window.classifier;
        if (classifier) {
            classifier.stopClassification();
        }
    }
});
