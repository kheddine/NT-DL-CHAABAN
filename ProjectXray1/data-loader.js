// data-loader.js

// Global variables to store training and test data
let mnistData = {
    train: { xs: null, ys: null },
    test: { xs: null, ys: null },
    validation: { xs: null, ys: null }
};

/**
 * Parse CSV file and convert to tensor data
 * CSV format: label (0-9) followed by 784 pixel values (0-255)
 * @param {File} file - CSV file containing MNIST data
 * @returns {Object} Object containing features (xs) and labels (ys) tensors
 */
async function loadCSVData(file) {
    if (!file || !file.name.endsWith('.csv')) {
        throw new Error('Please select a valid CSV file');
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            try {
                // Parse CSV content
                const content = e.target.result;
                const lines = content.split('\n').filter(line => line.trim() !== '');
                
                console.log(`Parsing ${lines.length} rows from ${file.name}`);
                
                // Arrays to store parsed data
                const labels = [];
                const pixels = [];
                
                // Parse each line
                for (let i = 0; i < lines.length; i++) {
                    const values = lines[i].split(',').map(val => val.trim());
                    
                    if (values.length !== 785) {
                        console.warn(`Skipping invalid row ${i}: expected 785 values, got ${values.length}`);
                        continue;
                    }
                    
                    // First value is the label (0-9)
                    const label = parseInt(values[0]);
                    if (isNaN(label) || label < 0 || label > 9) {
                        console.warn(`Skipping row ${i}: invalid label ${values[0]}`);
                        continue;
                    }
                    
                    // Remaining 784 values are pixel data (0-255)
                    const pixelData = values.slice(1).map(val => {
                        const pixel = parseInt(val);
                        return isNaN(pixel) ? 0 : pixel;
                    });
                    
                    labels.push(label);
                    pixels.push(pixelData);
                }
                
                if (labels.length === 0) {
                    throw new Error('No valid data found in CSV file');
                }
                
                // Convert to tensors and preprocess
                tf.tidy(() => {
                    // Convert pixel data to tensor and normalize to [0,1]
                    const xs = tf.tensor2d(pixels, [labels.length, 784])
                        .div(255.0) // Normalize pixel values
                        .reshape([labels.length, 28, 28, 1]); // Reshape to image format
                    
                    // Convert labels to one-hot encoded tensor
                    const ys = tf.oneHot(labels, 10);
                    
                    resolve({ xs, ys });
                });
                
            } catch (error) {
                reject(new Error(`Error parsing CSV file: ${error.message}`));
            }
        };
        
        reader.onerror = () => reject(new Error('Failed to read CSV file'));
        reader.readAsText(file);
    });
}

/**
 * Load training data from uploaded CSV file
 * @param {File} file - Training CSV file
 * @returns {Object} Training data tensors
 */
async function loadTrainFromFiles(file) {
    console.log('Loading training data...');
    const data = await loadCSVData(file);
    mnistData.train = data;
    return data;
}

/**
 * Load test data from uploaded CSV file
 * @param {File} file - Test CSV file
 * @returns {Object} Test data tensors
 */
async function loadTestFromFiles(file) {
    console.log('Loading test data...');
    const data = await loadCSVData(file);
    mnistData.test = data;
    return data;
}

/**
 * Split training data into training and validation sets
 * @param {tf.Tensor} xs - Features tensor
 * @param {tf.Tensor} ys - Labels tensor
 * @param {number} valRatio - Ratio of data to use for validation (default: 0.1)
 * @returns {Object} Split data {trainXs, trainYs, valXs, valYs}
 */
function splitTrainVal(xs, ys, valRatio = 0.1) {
    return tf.tidy(() => {
        const numSamples = xs.shape[0];
        const numVal = Math.floor(numSamples * valRatio);
        const numTrain = numSamples - numVal;
        
        // Create shuffled indices
        const indices = tf.util.createShuffledIndices(numSamples);
        
        // Split indices into training and validation
        const trainIndices = indices.slice(0, numTrain);
        const valIndices = indices.slice(numTrain);
        
        // Split data using indices
        const trainXs = tf.gather(xs, trainIndices);
        const trainYs = tf.gather(ys, trainIndices);
        const valXs = tf.gather(xs, valIndices);
        const valYs = tf.gather(ys, valIndices);
        
        return { trainXs, trainYs, valXs, valYs };
    });
}

/**
 * Get a random batch of test samples for preview
 * @param {tf.Tensor} xs - Test features tensor
 * @param {tf.Tensor} ys - Test labels tensor
 * @param {number} k - Number of samples to return (default: 5)
 * @returns {Object} Batch data {samples, labels, indices}
 */
function getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
        const numSamples = xs.shape[0];
        const indices = [];
        
        // Generate k unique random indices
        while (indices.length < k) {
            const idx = Math.floor(Math.random() * numSamples);
            if (!indices.includes(idx)) {
                indices.push(idx);
            }
        }
        
        // Gather the selected samples
        const samples = tf.gather(xs, indices);
        const labels = tf.gather(ys, indices);
        
        return { samples, labels, indices };
    });
}

/**
 * Draw a 28x28 grayscale image to a canvas element
 * @param {tf.Tensor} tensor - Image tensor [28,28,1] or [1,28,28,1]
 * @param {HTMLCanvasElement} canvas - Canvas element to draw on
 * @param {number} scale - Scale factor for display (default: 4)
 */
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    tf.tidy(() => {
        // Ensure tensor is 2D [28,28]
        let imageTensor = tensor.squeeze(); // Remove singleton dimensions
        
        // Rescale pixel values to 0-255 for display
        imageTensor = imageTensor.mul(255).cast('int32');
        
        // Get image data
        const data = imageTensor.dataSync();
        
        // Set canvas dimensions
        canvas.width = 28 * scale;
        canvas.height = 28 * scale;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw each pixel scaled up
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const pixelValue = data[y * 28 + x];
                const color = `rgb(${pixelValue}, ${pixelValue}, ${pixelValue})`;
                
                ctx.fillStyle = color;
                ctx.fillRect(x * scale, y * scale, scale, scale);
            }
        }
    });
}

/**
 * Get data statistics for display
 * @returns {Object} Data statistics
 */
function getDataStats() {
    const stats = {
        trainSamples: mnistData.train.xs ? mnistData.train.xs.shape[0] : 0,
        testSamples: mnistData.test.xs ? mnistData.test.xs.shape[0] : 0,
        validationSamples: mnistData.validation.xs ? mnistData.validation.xs.shape[0] : 0
    };
    
    return stats;
}

/**
 * Clean up tensor memory
 */
function disposeData() {
    if (mnistData.train.xs) mnistData.train.xs.dispose();
    if (mnistData.train.ys) mnistData.train.ys.dispose();
    if (mnistData.test.xs) mnistData.test.xs.dispose();
    if (mnistData.test.ys) mnistData.test.ys.dispose();
    if (mnistData.validation.xs) mnistData.validation.xs.dispose();
    if (mnistData.validation.ys) mnistData.validation.ys.dispose();
    
    mnistData = {
        train: { xs: null, ys: null },
        test: { xs: null, ys: null },
        validation: { xs: null, ys: null }
    };
}
