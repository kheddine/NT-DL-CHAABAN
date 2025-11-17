// enhanced-app.js - Enhanced version with better performance
class EnhancedRockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.animationId = null;
        this.lastInferenceTime = 0;
        this.inferenceInterval = 500; // Run inference every 500ms
        
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
        this.startBtn = document.getElementById('startBtn');
        this.classifyBtn = document.getElementById('classifyBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.fpsDiv = document.getElementById('fps');
        
        this.classNames = ['Rock', 'Paper', 'Scissors'];
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.classifyBtn.addEventListener('click', () => this.startClassification());
        this.stopBtn.addEventListener('click', () => this.stopClassification());
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !this.isClassifying) {
                this.startClassification();
            } else if (e.code === 'Escape' && this.isClassifying) {
                this.stopClassification();
            }
        });
    }
    
    async loadModel() {
        try {
            this.updateStatus('Loading TensorFlow.js model...');
            
            // Warm up TensorFlow.js
            await tf.ready();
            console.log('TensorFlow.js is ready');
            
            // Load the model - update the path to your actual model
            this.model = await tf.loadLayersModel('./model/model.json');
            
            // Warm up the model with a dummy prediction
            await this.warmUpModel();
            
            this.isModelLoaded = true;
            this.updateStatus('Model loaded successfully! Click "Start Camera" to begin.');
            console.log('Model loaded and warmed up');
            
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('Error loading model. Check console for details.');
        }
    }
    
    async warmUpModel() {
        const warmupTensor = tf.zeros([1, 224, 224, 3]);
        await this.model.predict(warmupTensor).data();
        warmupTensor.dispose();
    }
    
    async startCamera() {
        try {
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
                    resolve();
                };
            });
            
            this.video.play();
            this.startBtn.disabled = true;
            this.classifyBtn.disabled = false;
            this.updateStatus('Camera started. Click "Start Classification" to begin.');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.updateStatus('Error accessing camera. Please check permissions.');
        }
    }
    
    async startClassification() {
        if (!this.isModelLoaded) {
            alert('Model not loaded yet. Please wait.');
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
        }
    }
    
    captureAndPreprocessFrame() {
        return tf.tidy(() => {
            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, 224, 224);
            
            // Convert to tensor and preprocess
            return tf.browser.fromPixels(this.canvas)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims(0);
        });
    }
    
    processPredictions(predictions, inferenceTime) {
        const maxConfidence = Math.max(...predictions);
        const predictedClass = predictions.indexOf(maxConfidence);
        const className = this.classNames[predictedClass];
        
        // Update UI with results
        this.updateResults(className, predictions, inferenceTime);
    }
    
    updateResults(className, predictions, inferenceTime) {
        // Update main result
        const confidence = (predictions[this.classNames.indexOf(className)] * 100).toFixed(1);
        this.resultDiv.innerHTML = `
            <div>Prediction: <strong>${className}</strong></div>
            <div>Confidence: <strong>${confidence}%</strong></div>
            <div style="font-size: 0.8em; color: #666;">Inference: ${inferenceTime.toFixed(1)}ms</div>
        `;
        
        // Update confidence bars
        this.updateConfidenceBars(predictions);
    }
    
    updateConfidenceBars(predictions) {
        let confidenceHTML = '<div style="margin-top: 10px;">Confidence Levels:</div>';
        
        this.classNames.forEach((name, index) => {
            const confidence = predictions[index] * 100;
            const barWidth = Math.max(confidence, 5); // Minimum width for visibility
            
            confidenceHTML += `
                <div class="confidence-bar" style="margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${name}</span>
                        <span>${confidence.toFixed(1)}%</span>
                    </div>
                    <div style="background: #f0f0f0; border-radius: 3px; height: 20px;">
                        <div style="background: #007bff; width: ${barWidth}%; height: 100%; border-radius: 3px; transition: width 0.3s;"></div>
                    </div>
                </div>
            `;
        });
        
        this.confidenceDiv.innerHTML = confidenceHTML;
    }
    
    updateStatus(message) {
        this.resultDiv.textContent = message;
    }
}

// Initialize enhanced application
document.addEventListener('DOMContentLoaded', () => {
    new EnhancedRockPaperScissorsClassifier();
});
