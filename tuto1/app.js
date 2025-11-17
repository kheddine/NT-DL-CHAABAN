// app.js
class RockPaperScissorsClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isClassifying = false;
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultDiv = document.getElementById('result');
        this.confidenceDiv = document.getElementById('confidence');
        this.startBtn = document.getElementById('startBtn');
        this.classifyBtn = document.getElementById('classifyBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        this.classNames = ['Rock', 'Paper', 'Scissors'];
        
        this.initializeEventListeners();
        this.loadModel();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.classifyBtn.addEventListener('click', () => this.startClassification());
        this.stopBtn.addEventListener('click', () => this.stopClassification());
    }
    
    async loadModel() {
        try {
            this.resultDiv.textContent = 'Loading TensorFlow.js model...';
            
            // Load your converted TensorFlow.js model
            // Replace with your actual model path
            this.model = await tf.loadLayersModel('model/model.json');
            
            this.isModelLoaded = true;
            this.resultDiv.textContent = 'Model loaded successfully! Click "Start Camera" to begin.';
            console.log('Model loaded successfully');
            
        } catch (error) {
            console.error('Error loading model:', error);
            this.resultDiv.textContent = 'Error loading model. Check console for details.';
        }
    }
    
    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'user' 
                } 
            });
            
            this.video.srcObject = stream;
            this.startBtn.disabled = true;
            this.classifyBtn.disabled = false;
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.resultDiv.textContent = 'Error accessing camera. Please ensure you have a webcam connected and have granted permission.';
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
        
        this.classifyFrame();
    }
    
    stopClassification() {
        this.isClassifying = false;
        this.classifyBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.resultDiv.textContent = 'Classification stopped.';
        this.confidenceDiv.innerHTML = '';
    }
    
    async classifyFrame() {
        if (!this.isClassifying || !this.model) return;
        
        try {
            // Capture frame from video
            this.ctx.drawImage(this.video, 0, 0, 224, 224);
            const imageData = this.ctx.getImageData(0, 0, 224, 224);
            
            // Preprocess the image
            const tensor = this.preprocessImage(imageData);
            
            // Run inference
            const predictions = await this.model.predict(tensor).data();
            
            // Get the highest confidence prediction
            const maxConfidence = Math.max(...predictions);
            const predictedClass = predictions.indexOf(maxConfidence);
            
            // Display results
            this.displayResults(predictedClass, predictions);
            
            // Clean up
            tensor.dispose();
            
        } catch (error) {
            console.error('Error during classification:', error);
        }
        
        // Continue classification
        if (this.isClassifying) {
            requestAnimationFrame(() => this.classifyFrame());
        }
    }
    
    preprocessImage(imageData) {
        return tf.tidy(() => {
            // Convert ImageData to tensor
            let tensor = tf.browser.fromPixels(imageData)
                .resizeNearestNeighbor([224, 224])
                .toFloat();
            
            // Normalize pixel values to [0, 1]
            tensor = tensor.div(255.0);
            
            // Add batch dimension
            tensor = tensor.expandDims(0);
            
            return tensor;
        });
    }
    
    displayResults(predictedClass, predictions) {
        const className = this.classNames[predictedClass];
        const confidence = predictions[predictedClass];
        
        this.resultDiv.textContent = `Prediction: ${className} (${(confidence * 100).toFixed(2)}%)`;
        
        // Display confidence for all classes
        let confidenceHTML = '<div>Confidences:</div>';
        this.classNames.forEach((name, index) => {
            const conf = (predictions[index] * 100).toFixed(2);
            confidenceHTML += `<div class="confidence">${name}: ${conf}%</div>`;
        });
        
        this.confidenceDiv.innerHTML = confidenceHTML;
    }
}

// Alternative model loading function for different model formats
async function loadModelAlternative() {
    try {
        // If you have a different model format, you can use:
        // Option 1: Graph model
        // const model = await tf.loadGraphModel('model/model.json');
        
        // Option 2: Using tf.loadLayersModel for Keras models
        const model = await tf.loadLayersModel('path/to/your/model.json');
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        return null;
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new RockPaperScissorsClassifier();
});
