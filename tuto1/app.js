// Rock Paper Scissors Classifier - Fixed Version
class RockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.animationId = null;
        this.inferenceInterval = 400;
        
        this.classNames = ['Rock 👊', 'Paper 🖐️', 'Scissors ✌️'];
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

        document.addEventListener('visibilitychange', () => {
            if (document.hidden) this.stopClassification();
        });

        window.addEventListener('beforeunload', () => this.cleanup());
    }

    async loadModel() {
        try {
            this.updateStatus('🚀 Loading TensorFlow.js...');
            
            await tf.ready();
            const backend = tf.getBackend();
            this.updateStatus(`✅ TensorFlow.js loaded (${backend})`);
            
            console.log('Loading model from ./model/model.json');
            
            // Try multiple model paths
            const modelPaths = [
                './model/model.json',
                'model/model.json',
                '/model/model.json'
            ];
            
            let modelLoaded = false;
            for (const path of modelPaths) {
                try {
                    console.log('Trying path:', path);
                    this.model = await tf.loadLayersModel(path);
                    console.log('✅ Model loaded from:', path);
                    modelLoaded = true;
                    break;
                } catch (pathError) {
                    console.log('❌ Failed from:', path, pathError);
                    continue;
                }
            }
            
            if (!modelLoaded) {
                throw new Error('Could not load model from any path');
            }
            
            // Check model compatibility
            const compatible = await this.checkModelCompatibility();
            if (!compatible) {
                throw new Error('Model compatibility check failed');
            }
            
            this.isModelLoaded = true;
            this.updateStatus('✅ Model loaded! Click "Start Camera"');
            this.updateResult('🎯 Ready to classify');

        } catch (error) {
            console.error('Model loading error:', error);
            this.updateStatus('❌ Model error: ' + error.message);
            await this.createDemoModel();
        }
    }

    async checkModelCompatibility() {
        if (!this.model) return false;
        
        try {
            console.log('=== Model Compatibility Check ===');
            console.log('Input shape:', this.model.inputs[0].shape);
            console.log('Output shape:', this.model.outputs[0].shape);
            
            // Test prediction with correct input shape
            const inputShape = this.model.inputs[0].shape;
            const [batch, height, width, channels] = inputShape;
            
            const testInput = tf.zeros(inputShape);
            const testOutput = this.model.predict(testInput);
            await testOutput.data(); // This will throw error if incompatible
            
            console.log('✅ Model compatibility check passed');
            
            testInput.dispose();
            testOutput.dispose();
            return true;
            
        } catch (error) {
            console.error('❌ Model compatibility check failed:', error);
            return false;
        }
    }

    async createDemoModel() {
        try {
            this.updateStatus('🔧 Creating demo model...');
            
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
            this.updateStatus('⚠️ Demo model created');
            this.updateResult('🎮 Demo mode');
            
        } catch (error) {
            this.updateStatus('❌ Failed to create demo model');
        }
    }

    async startCamera() {
        try {
            this.updateStatus('📷 Requesting camera access...');
            
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
            this.updateStatus('✅ Camera ready!');

        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('❌ Camera error: ' + error.message);
            this.startBtn.disabled = false;
        }
    }

    async startClassification() {
        if (!this.isModelLoaded) {
            this.updateStatus('❌ Model not loaded');
            return;
        }

        this.isClassifying = true;
        this.classifyBtn.disabled = true;
        this.stopBtn.disabled = false;

        this.updateStatus('🔍 Classifying...');
        this.classifyFrame();
    }

    stopClassification() {
        this.isClassifying = false;
        this.classifyBtn.disabled = false;
        this.stopBtn.disabled = true;

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        this.updateStatus('⏹️ Classification stopped');
    }

    async classifyFrame() {
        if (!this.isClassifying) return;

        try {
            await this.runInference();
        } catch (error) {
            console.error('Classification error:', error);
        }

        this.animationId = requestAnimationFrame(() => this.classifyFrame());
    }

    async runInference() {
        try {
            const tensor = this.captureAndPreprocessFrame();
            
            // Verify tensor shape matches model input
            const inputShape = this.model.inputs[0].shape;
            const tensorShape = tensor.shape;
            
            if (tensorShape[1] !== inputShape[1] || tensorShape[2] !== inputShape[2]) {
                throw new Error(`Shape mismatch: Model expects [${inputShape}], got [${tensorShape}]`);
            }
            
            const predictions = await this.model.predict(tensor).data();
            tensor.dispose();
            
            this.processPredictions(predictions);

        } catch (error) {
            console.error('Inference error:', error);
            this.updateStatus('❌ Inference: ' + error.message);
        }
    }

    captureAndPreprocessFrame() {
        return tf.tidy(() => {
            // Clear canvas and draw video frame
            this.ctx.clearRect(0, 0, 150, 150);
            this.ctx.drawImage(this.video, 0, 0, 150, 150);
            
            // Get image data and convert to tensor
            const imageData = this.ctx.getImageData(0, 0, 150, 150);
            let tensor = tf.browser.fromPixels(imageData, 3); // 3 channels RGB
            
            // Ensure correct shape and type
            tensor = tensor.toFloat().div(255.0);
            
            // Add batch dimension - shape should be [1, 150, 150, 3]
            tensor = tensor.expandDims(0);
            
            return tensor;
        });
    }

    processPredictions(predictions) {
        if (!predictions || predictions.length !== 3) {
            console.error('Invalid predictions:', predictions);
            return;
        }

        const maxConfidence = Math.max(...predictions);
        const predictedClass = predictions.indexOf(maxConfidence);
        const className = this.classNames[predictedClass];
        const confidence = (maxConfidence * 100).toFixed(1);

        if (maxConfidence > 0.6) {
            this.updateResult(`${className} - ${confidence}%`);
            this.resultDiv.classList.add('pulse');
            setTimeout(() => this.resultDiv.classList.remove('pulse'), 600);
        } else {
            this.updateResult('🤔 Show clearer gesture');
        }

        this.updateConfidenceBars(predictions);
    }

    updateConfidenceBars(predictions) {
        let html = '<div class="confidence-title">Confidence:</div>';
        
        this.classNames.forEach((name, index) => {
            const confidence = (predictions[index] * 100).toFixed(1);
            const displayName = name.split(' ')[0];
            html += `
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
        
        this.confidenceDiv.innerHTML = html;
    }

    updateStatus(message) {
        if (this.statusDiv) this.statusDiv.textContent = message;
    }

    updateResult(message) {
        if (this.resultDiv) this.resultDiv.textContent = message;
    }

    cleanup() {
        this.stopClassification();
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }
        if (this.model) this.model.dispose();
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.classifier = new RockPaperScissorsClassifier();
});
