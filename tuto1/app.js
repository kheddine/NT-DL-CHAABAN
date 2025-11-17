// Rock Paper Scissors Classifier - TensorFlow.js
class RockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.animationId = null;
        this.inferenceInterval = 400; // ms between inferences
        
        this.classNames = ['Rock 👊', 'Paper 🖐️', 'Scissors ✌️'];
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

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.stopClassification();
            }
        });

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    async loadModel() {
        try {
            this.updateStatus('🚀 Loading TensorFlow.js...');
            this.updateResult('Initializing...');

            // Wait for TensorFlow.js to be ready
            await tf.ready();
            const backend = tf.getBackend();
            this.updateStatus(`✅ TensorFlow.js loaded (${backend})`);
            
            console.log('TensorFlow.js backend:', backend);

            // Load the trained model
            this.updateStatus('📦 Loading trained model...');
            this.model = await tf.loadLayersModel('./model/model.json');
            
            this.isModelLoaded = true;
            this.updateStatus('✅ Model loaded successfully! Click "Start Camera" to begin.');
            this.updateResult('🎯 Ready to classify');
            
            console.log('Trained model loaded successfully');

            // Enable classify button if camera is already running
            if (this.video.srcObject) {
                this.classifyBtn.disabled = false;
            }

        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('❌ Error loading model: ' + error.message);
            this.updateResult('😞 Load failed - check console');
            
            // Try to create a fallback demo model
            await this.createDemoModel();
        }
    }

    async createDemoModel() {
        try {
            this.updateStatus('🔧 Creating demo model...');
            
            // Simple demo model (untrained)
            const model = tf.sequential({
                layers: [
                    tf.layers.flatten({inputShape: [150, 150, 3]}),
                    tf.layers.dense({units: 64, activation: 'relu'}),
                    tf.layers.dense({units: 3, activation: 'softmax'})
                ]
            });
            
            model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.model = model;
            this.isModelLoaded = true;
            this.updateStatus('⚠️ Demo model created (not trained). Click "Start Camera".');
            this.updateResult('🎮 Demo mode - Add trained model');
            
        } catch (error) {
            this.updateStatus('❌ Failed to create demo model');
        }
    }

    async startCamera() {
        try {
            this.updateStatus('📷 Requesting camera access...');
            
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
            this.classifyBtn.disabled = !this.isModelLoaded;
            this.updateStatus('✅ Camera started! ' + 
                (this.isModelLoaded ? 'Click "Start Classification".' : 'Waiting for model...'));

        } catch (error) {
            console.error('Camera error:', error);
            let errorMessage = '❌ Camera error: ';
            
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Permission denied. Please allow camera access.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found. Please connect a camera.';
            } else if (error.name === 'NotSupportedError') {
                errorMessage += 'Browser not supported. Try Chrome or Firefox.';
            } else {
                errorMessage += error.message;
            }
            
            this.updateStatus(errorMessage);
            this.startBtn.disabled = false;
        }
    }

    async startClassification() {
        if (!this.isModelLoaded) {
            this.updateStatus('❌ Model not loaded yet. Please wait.');
            return;
        }

        if (!this.video.srcObject) {
            this.updateStatus('❌ Please start camera first.');
            return;
        }

        this.isClassifying = true;
        this.classifyBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.lastInferenceTime = 0;

        this.updateStatus('🔍 Classifying... Show rock, paper, or scissors!');
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

        this.updateStatus('⏹️ Classification stopped.');
        this.confidenceDiv.innerHTML = '';
    }

    async classifyFrame() {
        if (!this.isClassifying) return;

        const now = Date.now();
        
        // Throttle inference to maintain performance
        if (!this.lastInferenceTime || now - this.lastInferenceTime >= this.inferenceInterval) {
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
            this.updateStatus('❌ Inference error');
        }
    }

    captureAndPreprocessFrame() {
        return tf.tidy(() => {
            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, 150, 150);
            
            // Convert to tensor and preprocess (matches training)
            return tf.browser.fromPixels(this.canvas)
                .resizeNearestNeighbor([150, 150])
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
        const confidence = (predictions[this.classNames.indexOf(className)] * 100).toFixed(1);
        
        // Only show confident predictions
        if (maxConfidence > 0.7) {
            this.updateResult(`${className} - ${confidence}% confident`);
            this.resultDiv.classList.add('pulse');
            setTimeout(() => this.resultDiv.classList.remove('pulse'), 600);
        } else {
            this.updateResult('🤔 Show clearer hand gesture');
        }

        // Update confidence display
        this.updateConfidenceBars(predictions);
    }

    updateConfidenceBars(predictions) {
        let confidenceHTML = '<div class="confidence-title">Confidence Levels:</div>';
        
        this.classNames.forEach((name, index) => {
            const confidence = (predictions[index] * 100).toFixed(1);
            const displayName = name.split(' ')[0]; // Remove emoji for display
            confidenceHTML += `
                <div class="confidence-bar">
                    <div class="confidence-label">
                        <span>${displayName}</span>
                        <span>${confidence}%</span>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
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
            this.video.srcObject = null;
        }
        
        // Clean up TensorFlow.js memory
        if (this.model) {
            this.model.dispose();
        }
    }
}

// Initialize application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.classifier = new RockPaperScissorsClassifier();
});

// Handle errors
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});
