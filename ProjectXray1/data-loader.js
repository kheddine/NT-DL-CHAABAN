// data-loader.js

// Global variables to store training data
let trainingData = {
    pneumoniaFiles: [],
    normalFiles: [],
    xs: null,
    ys: null
};

/**
 * Load and preprocess multiple images from file input
 * @param {FileList} fileList - List of image files
 * @returns {tf.Tensor} Stacked tensor of preprocessed images
 */
async function loadImagesFromFiles(fileList) {
    if (!fileList || fileList.length === 0) {
        throw new Error('No files provided');
    }

    const images = [];
    
    // Process each file sequentially to avoid memory issues
    for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        
        // Skip non-image files
        if (!file.type.startsWith('image/')) {
            console.warn(`Skipping non-image file: ${file.name}`);
            continue;
        }

        try {
            const imageTensor = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                
                reader.onload = (e) => {
                    tf.tidy(() => {
                        try {
                            // Create image element and load data
                            const img = new Image();
                            img.onload = () => {
                                // Convert to tensor and preprocess
                                const tensor = tf.browser.fromPixels(img)
                                    .resizeNearestNeighbor([128, 128])
                                    .toFloat()
                                    .div(255.0); // Normalize to [0,1]
                                
                                resolve(tensor);
                            };
                            img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
                            img.src = e.target.result;
                        } catch (error) {
                            reject(error);
                        }
                    });
                };
                
                reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
                reader.readAsDataURL(file);
            });

            images.push(imageTensor);
            
            // Allow UI to update between image processing
            if (i % 5 === 0) {
                await tf.nextFrame();
            }
            
        } catch (error) {
            console.error(`Error processing file ${file.name}:`, error);
        }
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
 * Prepare training data from pneumonia and normal image files
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

    // Load pneumonia images (label = 1)
    const pneumoniaTensors = pneumoniaFiles.length > 0 ? 
        await loadImagesFromFiles(pneumoniaFiles) : 
        null;

    // Load normal images (label = 0)
    const normalTensors = normalFiles.length > 0 ? 
        await loadImagesFromFiles(normalFiles) : 
        null;

    return tf.tidy(() => {
        let xs, ys;

        if (pneumoniaTensors && normalTensors) {
            // Both classes available - concatenate
            xs = tf.concat([pneumoniaTensors, normalTensors], 0);
            
            // Create labels: 1 for pneumonia, 0 for normal
            const pneumoniaLabels = tf.ones([pneumoniaTensors.shape[0], 1]);
            const normalLabels = tf.zeros([normalTensors.shape[0], 1]);
            ys = tf.concat([pneumoniaLabels, normalLabels], 0);
            
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

        // Shuffle the dataset - FIXED: Convert indices to tensor before gathering
        const numSamples = xs.shape[0];
        const shuffledIndices = tf.tensor1d(tf.util.createShuffledIndices(numSamples), 'int32');
        xs = tf.gather(xs, shuffledIndices);
        ys = tf.gather(ys, shuffledIndices);
        
        // Dispose the indices tensor
        shuffledIndices.dispose();

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
            tf.tidy(() => {
                const img = new Image();
                img.onload = () => {
                    try {
                        // Convert to tensor, resize, normalize, and add batch dimension
                        const tensor = tf.browser.fromPixels(img)
                            .resizeNearestNeighbor([128, 128])
                            .toFloat()
                            .div(255.0)
                            .expandDims(0); // Add batch dimension [1, 128, 128, 3]
                        
                        resolve(tensor);
                    } catch (error) {
                        reject(error);
                    }
                };
                img.onerror = () => reject(new Error('Failed to load test image'));
                img.src = e.target.result;
            });
        };
        
        reader.onerror = () => reject(new Error('Failed to read test file'));
        reader.readAsDataURL(file);
    });
}

/**
 * Split training data into training and validation sets - FIXED VERSION
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
        
        // Create shuffled indices as tensor
        const indices = tf.tensor1d(tf.util.createShuffledIndices(numSamples), 'int32');
        
        // Split indices into training and validation
        const trainIndices = indices.slice(0, numTrain);
        const valIndices = indices.slice(numTrain);
        
        // Split data using indices
        const trainXs = tf.gather(xs, trainIndices);
        const trainYs = tf.gather(ys, trainIndices);
        const valXs = tf.gather(xs, valIndices);
        const valYs = tf.gather(ys, valIndices);
        
        // Clean up indices tensors
        indices.dispose();
        trainIndices.dispose();
        valIndices.dispose();
        
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
    const pneumoniaCount = tf.tidy(() => 
        trainingData.ys.sum().dataSync()[0]
    );
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
    if (trainingData.xs) trainingData.xs.dispose();
    if (trainingData.ys) trainingData.ys.dispose();
    
    trainingData = {
        pneumoniaFiles: [],
        normalFiles: [],
        xs: null,
        ys: null
    };
}
