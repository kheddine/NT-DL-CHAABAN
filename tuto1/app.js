// Enhanced Rock Paper Scissors Classifier - Improved Accuracy Version
class RockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.animationId = null;
        this.inferenceInterval = 300; // Reduced for faster response
        
        this.classNames = ['Rock 👊', 'Paper 🖐️', 'Scissors ✌️'];
        this.predictionHistory = [];
        this.historySize = 5; // Keep last 5 predictions for smoothing
        
        this.initializeElements();
        this.initializeEventListeners();
        this.loadModel();
    }

    initializeElements() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.processCanvas = document.createElement('canvas'); // Separate canvas for processing
        this.processCtx = this.processCanvas.getContext('2d');
        this.ctx = this.canvas.getContext('2d');
        this.resultDiv = document.getElementById('result');
        this.confidenceDiv = document.getElementById('confidence');
        this.statusDiv = document.getElementById('status');
        this.startBtn = document.getElementById('startBtn');
        this.classifyBtn = document.getElementById('classifyBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        // Set up processing canvas
        this.processCanvas.width = 224; // Increased resolution for better accuracy
        this.processCanvas.height = 224;
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
            await this.createEnhancedDemoModel();
        }
    }

    async checkModelCompatibility() {
        if (!this.model) return false;
        
        try {
            console.log('=== Model Compatibility Check ===');
            console.log('Input shape:', this.model.inputs[0].shape);
            console.log('Output shape:', this.model.outputs[0].shape);
            
            const inputShape = this.model.inputs[0].shape;
            const [batch, height, width, channels] = inputShape;
            
            const testInput = tf.zeros(inputShape);
            const testOutput = this.model.predict(testInput);
            await testOutput.data();
            
            console.log('✅ Model compatibility check passed');
            
            testInput.dispose();
            testOutput.dispose();
            return true;
            
        } catch (error) {
            console.error('❌ Model compatibility check failed:', error);
            return false;
        }
    }

    async createEnhancedDemoModel() {
        try {
            this.updateStatus('🔧 Creating enhanced demo model...');
            
            // More sophisticated demo model
            const model = tf.sequential({
                layers: [
                    tf.layers.conv2d({
                        inputShape: [224, 224, 3],
                        filters: 32,
                        kernelSize: 3,
                        activation: 'relu'
                    }),
                    tf.layers.maxPooling2d({ poolSize: 2 }),
                    
                    tf.layers.conv2d({
                        filters: 64,
                        kernelSize: 3,
                        activation: 'relu'
                    }),
                    tf.layers.maxPooling2d({ poolSize: 2 }),
                    
                    tf.layers.conv2d({
                        filters: 64,
                        kernelSize: 3,
                        activation: 'relu'
                    }),
                    tf.layers.maxPooling2d({ poolSize: 2 }),
                    
                    tf.layers.flatten(),
                    tf.layers.dense({ units: 128, activation: 'relu' }),
                    tf.layers.dropout({ rate: 0.5 }),
                    tf.layers.dense({ units: 3, activation: 'softmax' })
                ]
            });
            
            model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.model = model;
            this.isModelLoaded = true;
            this.updateStatus('⚠️ Enhanced demo model created');
            this.updateResult('🎮 Demo mode - Add trained model for better accuracy');
            
        } catch (error) {
            this.updateStatus('❌ Failed to create demo model');
        }
    }

    async startCamera() {
        try {
            this.updateStatus('📷 Requesting camera access...');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 1280 }, // Higher resolution for better quality
                    height: { ideal: 720 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
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
            this.updateStatus('✅ Camera ready! Position hand clearly in frame');

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
        this.predictionHistory = []; // Clear history

        this.updateStatus('🔍 Classifying... Show clear hand gesture');
        this.classifyFrame();
    }

    stopClassification() {
        this.isClassifying = false;
        this.classifyBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.predictionHistory = [];

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        this.updateStatus('⏹️ Classification stopped');
        this.confidenceDiv.innerHTML = '';
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
            
            const inputShape = this.model.inputs[0].shape;
            const tensorShape = tensor.shape;
            
            if (tensorShape[1] !== inputShape[1] || tensorShape[2] !== inputShape[2]) {
                throw new Error(`Shape mismatch: Model expects [${inputShape}], got [${tensorShape}]`);
            }
            
            const predictions = await this.model.predict(tensor).data();
            tensor.dispose();
            
            this.processPredictionsWithSmoothing(predictions);

        } catch (error) {
            console.error('Inference error:', error);
            this.updateStatus('❌ Inference: ' + error.message);
        }
    }

    captureAndPreprocessFrame() {
        return tf.tidy(() => {
            // Use higher resolution processing canvas
            this.processCtx.clearRect(0, 0, 224, 224);
            
            // Calculate aspect ratio and center the crop
            const videoAspect = this.video.videoWidth / this.video.videoHeight;
            const targetAspect = 224 / 224;
            
            let drawWidth, drawHeight, offsetX, offsetY;
            
            if (videoAspect > targetAspect) {
                // Video is wider
                drawHeight = 224;
                drawWidth = this.video.videoWidth * (224 / this.video.videoHeight);
                offsetX = (224 - drawWidth) / 2;
                offsetY = 0;
            } else {
                // Video is taller
                drawWidth = 224;
                drawHeight = this.video.videoHeight * (224 / this.video.videoWidth);
                offsetX = 0;
                offsetY = (224 - drawHeight) / 2;
            }
            
            // Draw video frame centered and cropped
            this.processCtx.drawImage(
                this.video, 
                offsetX, offsetY, drawWidth, drawHeight
            );
            
            // Apply image enhancement
            this.applyImageEnhancement();
            
            // Convert to tensor
            const imageData = this.processCtx.getImageData(0, 0, 224, 224);
            let tensor = tf.browser.fromPixels(imageData, 3);
            
            // Enhanced preprocessing
            tensor = this.enhancedPreprocessing(tensor);
            
            return tensor.expandDims(0);
        });
    }

    applyImageEnhancement() {
        const imageData = this.processCtx.getImageData(0, 0, 224, 224);
        const data = imageData.data;
        
        // Simple contrast enhancement
        for (let i = 0; i < data.length; i += 4) {
            // Increase contrast
            data[i] = this.clamp(data[i] * 1.1);     // Red
            data[i + 1] = this.clamp(data[i + 1] * 1.1); // Green
            data[i + 2] = this.clamp(data[i + 2] * 1.1); // Blue
        }
        
        this.processCtx.putImageData(imageData, 0, 0);
    }

    clamp(value) {
        return Math.max(0, Math.min(255, value));
    }

    enhancedPreprocessing(tensor) {
        return tf.tidy(() => {
            // Convert to float and normalize
            tensor = tensor.toFloat().div(255.0);
            
            // Optional: Apply additional preprocessing
            // Uncomment if your model was trained with specific preprocessing
            /*
            // Standardization (if model was trained this way)
            const mean = tensor.mean();
            const std = tensor.std().add(tf.scalar(1e-7));
            tensor = tensor.sub(mean).div(std);
            */
            
            return tensor;
        });
    }

    processPredictionsWithSmoothing(predictions) {
        if (!predictions || predictions.length !== 3) {
            console.error('Invalid predictions:', predictions);
            return;
        }

        // Add current prediction to history
        this.predictionHistory.push([...predictions]);
        
        // Keep only the last N predictions
        if (this.predictionHistory.length > this.historySize) {
            this.predictionHistory.shift();
        }

        // Apply temporal smoothing - average last N predictions
        const smoothedPredictions = this.smoothPredictions();
        
        const maxConfidence = Math.max(...smoothedPredictions);
        const predictedClass = smoothedPredictions.indexOf(maxConfidence);
        const className = this.classNames[predictedClass];
        const confidence = (maxConfidence * 100).toFixed(1);

        // Dynamic confidence threshold based on prediction stability
        const stability = this.calculatePredictionStability();
        const confidenceThreshold = 0.6 - (stability * 0.2); // Lower threshold if stable

        if (maxConfidence > confidenceThreshold && stability > 0.3) {
            this.updateResult(`${className} - ${confidence}%`);
            this.showPredictionTips(className);
        } else {
            this.updateResult('🤔 Show clearer hand gesture');
            this.showGestureTips();
        }

        this.updateConfidenceBars(smoothedPredictions, stability);
    }

    smoothPredictions() {
        if (this.predictionHistory.length === 0) return [0, 0, 0];
        
        const smoothed = [0, 0, 0];
        const historyLength = this.predictionHistory.length;
        
        // Weight recent predictions more heavily
        for (let i = 0; i < historyLength; i++) {
            const weight = (i + 1) / historyLength; // Linear weighting
            const predictions = this.predictionHistory[i];
            
            for (let j = 0; j < 3; j++) {
                smoothed[j] += predictions[j] * weight;
            }
        }
        
        // Normalize
        const sum = smoothed.reduce((a, b) => a + b, 0);
        return smoothed.map(val => val / sum);
    }

    calculatePredictionStability() {
        if (this.predictionHistory.length < 2) return 0;
        
        let stability = 0;
        for (let i = 1; i < this.predictionHistory.length; i++) {
            const current = this.predictionHistory[i];
            const previous = this.predictionHistory[i - 1];
            
            // Calculate similarity between consecutive predictions
            let similarity = 0;
            for (let j = 0; j < 3; j++) {
                similarity += 1 - Math.abs(current[j] - previous[j]);
            }
            stability += similarity / 3;
        }
        
        return stability / (this.predictionHistory.length - 1);
    }

    showPredictionTips(className) {
        const tips = {
            'Rock 👊': '✅ Good! Keep fist tight and visible',
            'Paper 🖐️': '✅ Good! Keep fingers straight and spread',
            'Scissors ✌️': '✅ Good! Clear V-shape with fingers'
        };
        
        // Only show tips occasionally to avoid distraction
        if (Math.random() < 0.1) { // 10% chance
            this.updateStatus(tips[className]);
        }
    }

    showGestureTips() {
        const tips = [
            '💡 Tip: Make sure your hand fills most of the frame',
            '💡 Tip: Use good lighting without shadows',
            '💡 Tip: Keep background simple and uncluttered',
            '💡 Tip: Hold gesture steady for 2-3 seconds'
        ];
        
        // Rotate through tips
        const randomTip = tips[Math.floor(Math.random() * tips.length)];
        this.updateStatus(randomTip);
    }

    updateConfidenceBars(predictions, stability) {
        let html = `
            <div class="confidence-title">
                Confidence (Stability: ${(stability * 100).toFixed(0)}%)
            </div>
        `;
        
        this.classNames.forEach((name, index) => {
            const confidence = (predictions[index] * 100).toFixed(1);
            const displayName = name.split(' ')[0];
            const isTopPrediction = predictions[index] === Math.max(...predictions);
            
            html += `
                <div class="confidence-bar ${isTopPrediction ? 'top-prediction' : ''}">
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
        if (this.resultDiv) {
            this.resultDiv.textContent = message;
            this.resultDiv.classList.add('pulse');
            setTimeout(() => this.resultDiv.classList.remove('pulse'), 600);
        }
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
