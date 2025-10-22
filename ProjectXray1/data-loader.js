// data-loader.js

// Global variables to store training data
let trainingData = {
    pneumoniaFiles: [],
    normalFiles: [],
    xs: null,
    ys: null
};

/**
 * Load and preprocess multiple images from file input - OPTIMIZED VERSION
 * @param {FileList} fileList - List of image files
 * @returns {tf.Tensor} Stacked tensor of preprocessed images
 */
async function loadImagesFromFiles(fileList) {
    if (!fileList || fileList.length === 0) {
        throw new Error('No files provided');
    }

    const images = [];
    const promises = [];
    
    // Process files in parallel batches for faster loading
    for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        
        // Skip non-image files
        if (!file.type.startsWith('image/')) {
            console.warn(`Skipping non-image file: ${file.name}`);
            continue;
        }

        const promise = new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    // Process image in one tidy block for memory efficiency
                    const tensor = tf.tidy(() => {
                        return tf.browser.fromPixels(img)
                            .resizeNearestNeighbor([128, 128])
                            .toFloat()
                            .div(255.0); // Normalize to [0,1]
                    });
                    resolve(tensor);
                };
                img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
                img.src = e.target.result;
            };
            
            reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
            reader.readAsDataURL(file);
        });

        promises.push(promise);
        
        // Process in batches of 4 to avoid memory issues but maintain speed
        if (promises.length >= 4) {
            const batchResults = await Promise.all(promises);
            images.push(...batchResults);
            promises.length = 0; // Clear the array
            await tf.nextFrame(); // Allow UI to update
        }
    }
    
    // Process any remaining promises
    if (promises.length > 0) {
        const remainingResults = await Promise.all(promises);
        images.push(...remainingResults);
    }

    if (images.length === 0) {
        throw new Error('No valid images were processed');
    }

    // Stack all image tensors into a single tensor
    return tf.tidy(() => {
        return tf.stack(images);
    });
}

/**
 * Prepare training data from pneumonia and normal image files - FIXED VERSION
 * @param {FileList} pneumoniaFiles - Pneumonia X-ray images
 * @param {FileList} normalFiles - Normal X-ray images
 * @returns {Object} Object containing features (xs) and labels (ys) tensors
 */
async function prepareTrainingData(pneumoniaFiles, normalFiles) {
    // Clear previous data
    if (trainingData.xs) {
        trainingData.xs.dispose();
        trainingData.ys.dispose();
    }

    console.log(`Loading ${pneumoniaFiles.length} pneumonia images and ${normalFiles.length} normal images`);

    // Load both datasets in parallel for faster loading
    const [pneumoniaTensors, normalTensors] = await Promise.all([
        pneumoniaFiles.length > 0 ? loadImagesFromFiles(pneumoniaFiles) : null,
        normalFiles.length > 0 ? loadImagesFromFiles(normalFiles) : null
    ]);

    return tf.tidy(() => {
        let xs, ys;

        if (pneumoniaTensors && normalTensors) {
            // Both classes available - concatenate
            xs = tf.concat([pneumoniaTensors, normalTensors], 0);
            
            // Create labels: 1 for pneumonia, 0 for normal
            const pneumoniaLabels = tf.ones([pneumoniaTensors.shape[0], 1]);
            const normalLabels = tf.zeros([normalTensors.shape[0], 1]);
            ys = tf.concat([pneumoniaLabels, normalLabels], 0);
            
            // Clean up individual tensors
            pneumoniaTensors.dispose();
            normalTensors.dispose();
            
        } else if (pneumoniaTensors) {
            // Only pneumonia images
            xs = pneumoniaTensors;
            ys = tf.ones([pneumoniaTensors.shape[0], 1]);
            
        } else if (normalTensors) {
            // Only normal images
            xs = normalTensors;
            ys = tf.zeros([normalTensors.shape[0], 1]);
            
        } else {
            throw new Error('No valid images found for training');
        }

        // Shuffle the dataset - FIXED: Convert Uint32Array to regular array first
        const numSamples = xs.shape[0];
        const shuffledIndicesArray = tf.util.createShuffledIndices(numSamples);
        
        // Convert Uint32Array to regular array to fix tensor1d issue
        const regularArray = Array.from(shuffledIndicesArray);
        const shuffledIndices = tf.tensor1d(regularArray, 'int32');
        
        const shuffledXs = tf.gather(xs, shuffledIndices);
        const shuffledYs = tf.gather(ys, shuffledIndices);
        
        // Clean up original tensors and indices
        if (xs !== pneumoniaTensors && xs !== normalTensors) {
            xs.dispose();
        }
        ys.dispose();
        shuffledIndices.dispose();

        return { xs: shuffledXs, ys: shuffledYs };
    });
}

/**
 * Load and preprocess a single test image for prediction - OPTIMIZED
 * @param {File} file - Single image file
 * @returns {tf.Tensor} Preprocessed image tensor ready for model prediction
 */
async function loadTestImage(file) {
    if (!file || !file.type.startsWith('image/')) {
        throw new Error('Please select a valid image file');
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const tensor = tf.tidy(() => {
                    return tf.browser.fromPixels(img)
                        .resizeNearestNeighbor([128, 128])
                        .toFloat()
                        .div(255.0)
                        .expandDims(0); // Add batch dimension [1, 128, 128, 3]
                });
                resolve(tensor);
            };
            img.onerror = () => reject(new Error('Failed to load test image'));
            img.src = e.target.result;
        };
        
        reader.onerror = () => reject(new Error('Failed to read test file'));
        reader.readAsDataURL(file);
    });
}

/**
 * Split training data into training and validation sets - SIMPLIFIED VERSION
 * @param {tf.Tensor} xs - Features tensor
 * @param {tf.Tensor} ys - Labels tensor
 * @param {number} valRatio - Ratio of data to use for validation (default: 0.2)
 * @returns {Object} Split data {trainXs, trainYs, valXs, valYs}
 */
function splitTrainVal(xs, ys, valRatio = 0.2) {
    return tf.tidy(() => {
        const numSamples = xs.shape[0];
        const numVal = Math.floor(numSamples * valRatio);
        const numTrain = numSamples - numVal;
        
        // Simple split without complex shuffling
        const trainXs = xs.slice(0, numTrain);
        const trainYs = ys.slice(0, numTrain);
        const valXs = xs.slice(numTrain);
        const valYs = ys.slice(numTrain);
        
        return { trainXs, trainYs, valXs, valYs };
    });
}

/**
 * Get current training data statistics
 * @returns {Object} Object containing counts and shape information
 */
function getTrainingDataStats() {
    if (!trainingData.xs) {
        return { pneumoniaCount: 0, normalCount: 0, totalSamples: 0 };
    }
    
    const totalSamples = trainingData.ys.shape[0];
    // Use dataSync only once for efficiency
    const ysData = trainingData.ys.dataSync();
    const pneumoniaCount = ysData.reduce((sum, val) => sum + val, 0);
    const normalCount = totalSamples - pneumoniaCount;
    
    return {
        pneumoniaCount: Math.round(pneumoniaCount),
        normalCount: Math.round(normalCount),
        totalSamples: totalSamples,
        inputShape: trainingData.xs.shape.slice(1)
    };
}

/**
 * Clean up tensor memory
 */
function disposeData() {
    if (trainingData.xs) {
        trainingData.xs.dispose();
        trainingData.xs = null;
    }
    if (trainingData.ys) {
        trainingData.ys.dispose();
        trainingData.ys = null;
    }
    
    trainingData.pneumoniaFiles = [];
    trainingData.normalFiles = [];
}
