// data-loader.js

// Global variables to store training data
let trainingData = {
    pneumoniaFiles: [],
    normalFiles: [],
    xs: null,
    ys: null
};

/**
 * Load and preprocess a single image file
 * @param {File} file - Image file
 * @returns {tf.Tensor} Preprocessed image tensor
 */
async function loadSingleImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const tensor = tf.tidy(() => {
                    return tf.browser.fromPixels(img)
                        .resizeNearestNeighbor([128, 128])
                        .toFloat()
                        .div(255.0);
                });
                resolve(tensor);
            };
            img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
            img.src = e.target.result;
        };
        
        reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
        reader.readAsDataURL(file);
    });
}

/**
 * Prepare training data from pneumonia and normal image files - SIMPLIFIED VERSION
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

    // Create file list with labels first (shuffle at file level)
    const filesWithLabels = [];
    
    // Add pneumonia files with label 1
    for (let i = 0; i < pneumoniaFiles.length; i++) {
        filesWithLabels.push({
            file: pneumoniaFiles[i],
            label: 1
        });
    }
    
    // Add normal files with label 0
    for (let i = 0; i < normalFiles.length; i++) {
        filesWithLabels.push({
            file: normalFiles[i],
            label: 0
        });
    }
    
    // Shuffle the file list (simple JavaScript array shuffle)
    for (let i = filesWithLabels.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [filesWithLabels[i], filesWithLabels[j]] = [filesWithLabels[j], filesWithLabels[i]];
    }

    const images = [];
    const labels = [];
    
    // Load images in batches for better performance
    const batchSize = 4;
    
    for (let i = 0; i < filesWithLabels.length; i += batchSize) {
        const batchEnd = Math.min(i + batchSize, filesWithLabels.length);
        const batchPromises = [];
        
        for (let j = i; j < batchEnd; j++) {
            batchPromises.push(loadSingleImage(filesWithLabels[j].file));
        }
        
        const batchTensors = await Promise.all(batchPromises);
        images.push(...batchTensors);
        
        // Add labels for this batch
        for (let j = i; j < batchEnd; j++) {
            labels.push(filesWithLabels[j].label);
        }
        
        // Allow UI to update
        if (i % 8 === 0) {
            await tf.nextFrame();
        }
    }

    if (images.length === 0) {
        throw new Error('No valid images were processed');
    }

    // Convert to tensors
    return tf.tidy(() => {
        const xs = tf.stack(images);
        const ys = tf.tensor2d(labels, [labels.length, 1]);
        
        return { xs, ys };
    });
}

/**
 * Load and preprocess a single test image for prediction
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
 * Split training data into training and validation sets - SIMPLE SLICE
 * @param {tf.Tensor} xs - Features tensor
 * @param {tf.Tensor} ys - Labels tensor
 * @param {number} valRatio - Ratio of data to use for validation (default: 0.2)
 * @returns {Object} Split data {trainXs, trainYs, valXs, valYs}
 */
function splitTrainVal(xs, ys, valRatio = 0.2) {
    const numSamples = xs.shape[0];
    const numVal = Math.floor(numSamples * valRatio);
    const numTrain = numSamples - numVal;
    
    return {
        trainXs: xs.slice(0, numTrain),
        trainYs: ys.slice(0, numTrain),
        valXs: xs.slice(numTrain),
        valYs: ys.slice(numTrain)
    };
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
    const ysData = trainingData.ys.dataSync();
    let pneumoniaCount = 0;
    
    for (let i = 0; i < ysData.length; i++) {
        if (ysData[i] === 1) pneumoniaCount++;
    }
    
    const normalCount = totalSamples - pneumoniaCount;
    
    return {
        pneumoniaCount: pneumoniaCount,
        normalCount: normalCount,
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
