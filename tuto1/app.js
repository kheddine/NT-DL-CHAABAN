// app.js - Improved version with detailed error reporting
class RockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.animationId = null;
        this.lastInferenceTime = 0;
        this.inferenceInterval = 300;
        
        this.classNames = ['Rock', 'Paper', 'Scissors'];
        this.initializeElements();
        this.initializeEventListeners();
        this.loadModel();
    }

    initializeElements() {
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
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.classifyBtn.addEventListener('click', () => this.startClassification());
        this.stopBtn.addEventListener('click', () => this.stopClassification());
    }

    async loadModel() {
        try {
            this.updateStatus('Loading TensorFlow.js...');
            
            // Check if TensorFlow.js is available
            if (typeof tf === 'undefined') {
                throw new Error('TensorFlow.js not loaded. Check CDN connection.');
            }

            await tf.ready();
            console.log('TensorFlow.js backend:', tf.getBackend());
            
            this.updateStatus('Loading model files...');
            
            // Try different possible model paths
            const modelPaths = [
                './model/model.json',
                'model/model.json',
                '/model/model.json',
                './tfjs_model/model.json'
            ];
            
            let modelLoaded = false;
            for (const modelPath of modelPaths) {
                try {
                    console.log('Trying to load model from:', modelPath);
                    this.model = await tf.loadLayersModel(modelPath);
                    console.log('Model loaded successfully from:', modelPath);
                    modelLoaded = true;
                    break;
                } catch (pathError) {
                    console.log('Failed to load from', modelPath, pathError);
                    continue;
                }
            }
            
            if (!modelLoaded) {
                throw new Error('Could not load model from any path. Check model files.');
            }

            // Test the model with a warm-up inference
            this.updateStatus('Warming up model...');
            await this.warmUpModel();
            
            this.isModelLoaded = true;
            this.updateStatus('Model loaded successfully! Click "Start Camera" to begin.');
            this.updateResult('Ready to classify');
            
        } catch (error) {
            console.error('Model loading failed:', error);
            this.updateStatus('Model loading failed: ' + error.message);
            this.updateResult('Load failed - check console');
            
            // Provide specific guidance based on error type
            if (error.message.includes('404')) {
                this.updateStatus('Model files not found. Check if model.json and .bin files exist in model/ folder.');
            } else if (error.message.includes('CORS')) {
                this.updateStatus('CORS error: Use a local web server, not file:// protocol.');
            }
        }
    }

    async warmUpModel() {
        const warmupTensor = tf.zeros([1, 224, 224, 3]);
        const result = this.model.predict(warmupTensor);
        await result.data();
        warmupTensor.dispose();
        result.dispose();
    }

    async startCamera() {
        try {
            this.updateStatus('Requesting camera access...');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 },
                    facingMode: 'user' 
                } 
            });
            
            this.video.srcObject = stream;
            
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play().then(resolve);
                };
            });

            this.startBtn.disabled = true;
            this.classifyBtn.disabled = !this.isModelLoaded;
            this.updateStatus('Camera started. ' + (this.isModelLoaded ? 'Click "Start Classification" to begin.' : 'Waiting for model...'));

        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('Camera error: ' + error.message);
        }
    }

    async startClassification() {
        if (!this.isModelLoaded) {
            this.updateStatus('Model not loaded yet. Please wait.');
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
        }

        this.updateStatus('Classification stopped.');
    }

    async classifyFrame() {
        if (!this.isClassifying) return;

        const now = Date.now();
        if (now - this.lastInferenceTime >= this.inferenceInterval) {
            await this.runInference();
            this.lastInferenceTime = now;
        }

        this.animationId = requestAnimationFrame(() => this.classifyFrame());
    }

    async runInference() {
        try {
            const tensor = this.captureAndPreprocessFrame();
            const predictions = await this.model.predict(tensor).data();
            this.processPredictions(predictions);
            tensor.dispose();
        } catch (error) {
            console.error('Inference error:', error);
        }
    }

    captureAndPreprocessFrame() {
        return tf.tidy(() => {
            this.ctx.drawImage(this.video, 0, 0, 224, 224);
            return tf.browser.fromPixels(this.canvas)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims(0);
        });
    }

    processPredictions(predictions) {
        let maxConfidence = 0;
        let predictedClass = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] > maxConfidence) {
                maxConfidence = predictions[i];
                predictedClass = i;
            }
        }

        const className = this.classNames[predictedClass];
        const confidence = (maxConfidence * 100).toFixed(1);
        
        this.updateResult(`${className} (${confidence}%)`);
        this.updateConfidenceBars(predictions);
    }

    updateConfidenceBars(predictions) {
        let html = '<div style="margin-bottom: 10px; font-weight: bold;">Confidence:</div>';
        this.classNames.forEach((name, i) => {
            const confidence = (predictions[i] * 100).toFixed(1);
            html += `
                <div class="confidence-bar">
                    <div class="confidence-label">
                        <span>${name}</span>
                        <span>${confidence}%</span>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: ${confidence}%;"></div>
                    </div>
                </div>
            `;
        });
        this.confidenceDiv.innerHTML = html;
    }

    updateStatus(message) {
        if (this.statusDiv) {
            this.statusDiv.textContent = message;
        }
        console.log('Status:', message);
    }

    updateResult(message) {
        if (this.resultDiv) {
            this.resultDiv.textContent = message;
        }
    }

    cleanup() {
        this.stopClassification();
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }
    }
}

// Initialize when ready
document.addEventListener('DOMContentLoaded', () => {
    window.classifier = new RockPaperScissorsClassifier();
});
