// data-loader.js
/**
 * Data loading and preprocessing utilities for MNIST CSV files
 * Handles file parsing, normalization, and tensor creation entirely in browser
 */

class MNISTDataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    /**
     * Parse CSV file and convert to tensor data
     * @param {File} file - CSV file containing MNIST data
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>} Normalized image and label tensors
     */
    async loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const csvText = event.target.result;
                    const { images, labels } = this.parseCSV(csvText);
                    
                    // Convert to tensors and normalize
                    const xs = tf.tensor4d(images, [images.length, 28, 28, 1])
                                .div(255.0); // Normalize pixel values to [0, 1]
                    
                    const ys = tf.oneHot(labels, 10); // Convert labels to one-hot encoding
                    
                    resolve({ xs, ys });
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    /**
     * Parse CSV text into arrays of images and labels
     * @param {string} csvText - Raw CSV content
     * @returns {{images: number[][], labels: number[]}} Parsed data
     */
    parseCSV(csvText) {
        const lines = csvText.split('\n');
        const images = [];
        const labels = [];
        
        for (const line of lines) {
            // Skip empty lines and potential headers
            if (!line.trim() || isNaN(parseInt(line[0]))) continue;
            
            const values = line.split(',').map(Number);
            const label = values[0];
            const pixels = values.slice(1, 785); // Get 784 pixel values
            
            // Validate data length
            if (pixels.length !== 784) {
                console.warn('Skipping invalid row with incorrect pixel count:', pixels.length);
                continue;
            }
            
            labels.push(label);
            images.push(pixels);
        }
        
        console.log(`Parsed ${images.length} samples with labels:`, 
                    Array.from(new Set(labels)).sort());
        
        return { images, labels };
    }

    /**
     * Load training data from file
     * @param {File} file - Training CSV file
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>} Training tensors
     */
    async loadTrainFromFiles(file) {
        this.trainData = await this.loadFromFile(file);
        return this.trainData;
    }

    /**
     * Load test data from file
     * @param {File} file - Test CSV file
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>} Test tensors
     */
    async loadTestFromFiles(file) {
        this.testData = await this.loadFromFile(file);
        return this.testData;
    }

    /**
     * Split training data into training and validation sets
     * @param {tf.Tensor} xs - Input features
     * @param {tf.Tensor} ys - Labels
     * @param {number} valRatio - Validation set ratio (default: 0.1)
     * @returns {Object} Split datasets
     */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        const numSamples = xs.shape[0];
        const numVal = Math.floor(numSamples * valRatio);
        const numTrain = numSamples - numVal;
        
        // Split indices
        const indices = tf.util.createShuffledIndices(numSamples);
        const trainIndices = indices.slice(0, numTrain);
        const valIndices = indices.slice(numTrain);
        
        // Create subsets using tidy to avoid memory leaks
        return tf.tidy(() => {
            const trainXs = xs.gather(trainIndices);
            const trainYs = ys.gather(trainIndices);
            const valXs = xs.gather(valIndices);
            const valYs = ys.gather(valIndices);
            
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    /**
     * Get random batch of test samples for preview
     * @param {tf.Tensor} xs - Test features
     * @param {tf.Tensor} ys - Test labels
     * @param {number} k - Number of samples (default: 5)
     * @returns {Object} Batch of test samples
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            
            // Generate k unique random indices
            while (indices.length < k) {
                const idx = Math.floor(Math.random() * numSamples);
                if (!indices.includes(idx)) indices.push(idx);
            }
            
            const batchXs = xs.gather(indices);
            const batchYs = ys.gather(indices);
            
            return {
                xs: batchXs,
                ys: batchYs,
                indices: indices
            };
        });
    }

    /**
     * Draw 28x28 tensor to canvas element
     * @param {tf.Tensor} tensor - Image tensor (shape: [28, 28, 1] or [1, 28, 28, 1])
     * @param {HTMLCanvasElement} canvas - Target canvas element
     * @param {number} scale - Scaling factor for display (default: 4)
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        tf.tidy(() => {
            // Ensure tensor is 2D and normalized
            const imageTensor = tensor.squeeze(); // Remove singleton dimensions
            const imageData = imageTensor.mul(255).cast('int32'); // Denormalize to 0-255
            
            // Set canvas dimensions
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.imageSmoothingEnabled = false;
            
            // Create image data for 28x28
            const imageArray = imageData.arraySync();
            const imgData = ctx.createImageData(28, 28);
            
            for (let i = 0; i < 28; i++) {
                for (let j = 0; j < 28; j++) {
                    const pixelValue = imageArray[i][j];
                    const idx = (i * 28 + j) * 4;
                    imgData.data[idx] = pixelValue;     // R
                    imgData.data[idx + 1] = pixelValue; // G
                    imgData.data[idx + 2] = pixelValue; // B
                    imgData.data[idx + 3] = 255;        // A
                }
            }
            
            // Draw original size to temp canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imgData, 0, 0);
            
            // Scale up to display canvas
            ctx.drawImage(tempCanvas, 0, 0, 28 * scale, 28 * scale);
        });
    }

    /**
     * Clean up tensors to prevent memory leaks
     */
    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
