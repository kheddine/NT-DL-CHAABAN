/**
 * Medical Image Data Loader and Preprocessor
 * Handles chest X-ray image loading, preprocessing, and augmentation
 * All processing happens client-side for privacy
 */

class MedicalImageLoader {
    constructor() {
        this.loadedImages = [];
        this.processedTensors = [];
        this.labels = [];
        this.trainingData = {
            normal: [],
            pneumonia: []
        };
        this.inputSize = [150, 150]; // Match model input size
        this.augmentationOptions = {
            rotation: true,
            flipping: true,
            brightness: true
        };
    }

    /**
     * Load training data from separate normal and pneumonia files
     */
    async loadTrainingData(normalFiles, pneumoniaFiles) {
        this.clearPreviousData();
        
        console.log(`Loading training data: ${normalFiles.length} normal, ${pneumoniaFiles.length} pneumonia images`);
        
        // Validate and load normal images
        const validNormalFiles = this.validateFiles(normalFiles);
        for (const file of validNormalFiles) {
            try {
                const imageData = await this.loadSingleImage(file);
                this.trainingData.normal.push(imageData);
            } catch (error) {
                console.warn(`Error loading normal image ${file.name}:`, error);
            }
        }
        
        // Validate and load pneumonia images
        const validPneumoniaFiles = this.validateFiles(pneumoniaFiles);
        for (const file of validPneumoniaFiles) {
            try {
                const imageData = await this.loadSingleImage(file);
                this.trainingData.pneumonia.push(imageData);
            } catch (error) {
                console.warn(`Error loading pneumonia image ${file.name}:`, error);
            }
        }
        
        // Combine all images for general use
        this.loadedImages = [
            ...this.trainingData.normal,
            ...this.trainingData.pneumonia
        ];
        
        console.log(`Training data loaded: ${this.trainingData.normal.length} normal, ${this.trainingData.pneumonia.length} pneumonia images`);
        return this.trainingData;
    }

    /**
     * Load test data from files
     */
    async loadTestData(files) {
        this.clearPreviousData();
        
        const validFiles = this.validateFiles(files);
        if (validFiles.length > 50) {
            throw new Error('Maximum 50 images allowed. Please select fewer files.');
        }

        // Load images sequentially to avoid memory issues
        for (let i = 0; i < validFiles.length; i++) {
            try {
                const imageData = await this.loadSingleImage(validFiles[i]);
                this.loadedImages.push(imageData);
                
                // Periodic cleanup
                if (i % 10 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            } catch (error) {
                console.error(`Error loading image ${validFiles[i].name}:`, error);
            }
        }

        console.log(`Test data loaded: ${this.loadedImages.length} images`);
        return this.loadedImages;
    }

    /**
     * Validate files for medical image processing
     */
    validateFiles(files) {
        return Array.from(files).filter(file => {
            const isValidType = file.type.startsWith('image/');
            const isValidSize = file.size <= 10 * 1024 * 1024; // 10MB limit
            
            if (!isValidType) {
                console.warn(`Skipping non-image file: ${file.name}`);
                return false;
            }
            if (!isValidSize) {
                console.warn(`Skipping large file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
                return false;
            }
            
            return true;
        });
    }

    /**
     * Load and process single medical image
     */
    async loadSingleImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            const img = new Image();
            
            reader.onload = (e) => {
                img.onload = () => {
                    // Validate image dimensions for medical imaging
                    if (img.width < 100 || img.height < 100) {
                        reject(new Error('Image dimensions too small for medical image analysis'));
                        return;
                    }
                    
                    if (img.width > 4096 || img.height > 4096) {
                        reject(new Error('Image dimensions too large for browser processing'));
                        return;
                    }
                    
                    resolve({
                        file: file,
                        element: img,
                        originalWidth: img.width,
                        originalHeight: img.height,
                        url: e.target.result
                    });
                };
                
                img.onerror = () => reject(new Error('Failed to load image'));
                img.src = e.target.result;
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    /**
     * Prepare training dataset from loaded files
     */
    prepareTrainingDataset() {
        if (this.trainingData.normal.length === 0 && this.trainingData.pneumonia.length === 0) {
            throw new Error('No training data loaded. Please load normal and pneumonia images first.');
        }

        const allTensors = [];
        const allLabels = [];
        
        // Process normal images
        console.log('Processing normal images...');
        this.trainingData.normal.forEach(imageData => {
            try {
                const tensor = this.preprocessSingleImage(imageData.element);
                allTensors.push(tensor);
                allLabels.push('Normal');
            } catch (error) {
                console.error('Error preprocessing normal image:', error);
            }
        });
        
        // Process pneumonia images
        console.log('Processing pneumonia images...');
        this.trainingData.pneumonia.forEach(imageData => {
            try {
                const tensor = this.preprocessSingleImage(imageData.element);
                allTensors.push(tensor);
                allLabels.push('Pneumonia');
            } catch (error) {
                console.error('Error preprocessing pneumonia image:', error);
            }
        });
        
        this.processedTensors = allTensors;
        this.labels = allLabels;
        
        console.log(`Training dataset prepared: ${allTensors.length} images (${this.trainingData.normal.length} normal, ${this.trainingData.pneumonia.length} pneumonia)`);
        
        return {
            tensors: allTensors,
            labels: allLabels
        };
    }

    /**
     * Preprocess images for inference
     */
    preprocessImages(images) {
        this.processedTensors = [];
        this.labels = []; // Clear labels for inference
        
        console.log('Preprocessing images for inference...');
        
        images.forEach((imageData, index) => {
            try {
                const tensor = this.preprocessSingleImage(imageData.element);
                this.processedTensors.push(tensor);
            } catch (error) {
                console.error(`Error preprocessing image ${index}:`, error);
            }
        });
        
        console.log(`Preprocessed ${this.processedTensors.length} images for inference`);
        return {
            tensors: this.processedTensors,
            labels: this.labels
        };
    }

    /**
     * Preprocess single image with medical imaging considerations
     */
    preprocessSingleImage(imgElement) {
        return tf.tidy(() => {
            // Convert image to tensor
            let tensor = tf.browser.fromPixels(imgElement);
            
            // Handle grayscale images (convert to RGB)
            if (tensor.shape[2] === 1) {
                tensor = tensor.concat([tensor, tensor], 2);
            } else if (tensor.shape[2] === 4) {
                // Remove alpha channel if present
                tensor = tensor.slice([0, 0, 0], [tensor.shape[0], tensor.shape[1], 3]);
            }
            
            // Resize to target size (150x150 for optimized performance)
            tensor = tf.image.resizeBilinear(tensor, this.inputSize);
            
            // Normalize pixel values to [0, 1] range
            tensor = tensor.div(255.0);
            
            // Medical imaging specific: enhance contrast for better feature extraction
            tensor = this.enhanceMedicalImageContrast(tensor);
            
            return tensor;
        });
    }

    /**
     * Contrast enhancement for medical images
     */
    enhanceMedicalImageContrast(tensor) {
        return tf.tidy(() => {
            // Simple contrast stretching
            const min = tensor.min();
            const max = tensor.max();
            const range = max.sub(min);
            
            // Avoid division by zero
            const safeRange = range.add(tf.scalar(1e-7));
            return tensor.sub(min).div(safeRange);
        });
    }

    /**
     * Data augmentation for training medical imaging models
     */
    augmentData(tensors, labels, augmentationCount = 2) {
        const augmentedTensors = [];
        const augmentedLabels = [];
        
        console.log(`Starting data augmentation: ${tensors.length} original images`);
        
        tensors.forEach((tensor, index) => {
            // Add original tensor
            augmentedTensors.push(tensor);
            augmentedLabels.push(labels[index]);
            
            // Create augmented versions
            for (let i = 0; i < augmentationCount; i++) {
                try {
                    const augmentedTensor = this.applyAugmentation(tensor);
                    augmentedTensors.push(augmentedTensor);
                    augmentedLabels.push(labels[index]);
                } catch (error) {
                    console.warn('Augmentation failed for one image:', error);
                }
            }
        });
        
        console.log(`Data augmentation applied: ${tensors.length} original → ${augmentedTensors.length} total`);
        
        return {
            tensors: augmentedTensors,
            labels: augmentedLabels
        };
    }

    /**
     * Apply random augmentations to medical image tensor
     */
    applyAugmentation(tensor) {
        return tf.tidy(() => {
            let augmented = tensor;
            
            // Random horizontal flip - common in medical imaging
            if (this.augmentationOptions.flipping && Math.random() > 0.5) {
                augmented = augmented.reverse(1); // Horizontal flip
            }
            
            // Random brightness adjustment
            if (this.augmentationOptions.brightness && Math.random() > 0.5) {
                const brightnessDelta = (Math.random() - 0.5) * 0.1; // ±5%
                augmented = tf.clipByValue(augmented.add(brightnessDelta), 0, 1);
            }
            
            // Random contrast adjustment
            if (this.augmentationOptions.brightness && Math.random() > 0.5) {
                const contrastFactor = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
                const mean = augmented.mean();
                augmented = augmented.sub(mean).mul(contrastFactor).add(mean);
                augmented = tf.clipByValue(augmented, 0, 1);
            }
            
            return augmented;
        });
    }

    /**
     * Split data into training and validation sets with stratification
     */
    splitData(tensors, labels, validationSplit = 0.2) {
        // Separate indices by class for stratified split
        const normalIndices = labels.map((label, idx) => label === 'Normal' ? idx : -1).filter(idx => idx !== -1);
        const pneumoniaIndices = labels.map((label, idx) => label === 'Pneumonia' ? idx : -1).filter(idx => idx !== -1);
        
        // Shuffle each class
        this.shuffleArray(normalIndices);
        this.shuffleArray(pneumoniaIndices);
        
        // Calculate validation size for each class
        const normalValSize = Math.floor(normalIndices.length * validationSplit);
        const pneumoniaValSize = Math.floor(pneumoniaIndices.length * validationSplit);
        
        // Create validation indices
        const valIndices = [
            ...normalIndices.slice(0, normalValSize),
            ...pneumoniaIndices.slice(0, pneumoniaValSize)
        ];
        
        // Create training indices
        const trainIndices = [
            ...normalIndices.slice(normalValSize),
            ...pneumoniaIndices.slice(pneumoniaValSize)
        ];
        
        // Shuffle final arrays
        this.shuffleArray(valIndices);
        this.shuffleArray(trainIndices);
        
        const trainTensors = trainIndices.map(i => tensors[i]);
        const trainLabels = trainIndices.map(i => labels[i]);
        const valTensors = valIndices.map(i => tensors[i]);
        const valLabels = valIndices.map(i => labels[i]);
        
        console.log(`Data split: ${trainTensors.length} training, ${valTensors.length} validation images`);
        
        return {
            train: { tensors: trainTensors, labels: trainLabels },
            validation: { tensors: valTensors, labels: valLabels }
        };
    }

    /**
     * Convert labels to one-hot encoding for categorical classification
     */
    labelsToOneHot(labels) {
        const classNames = ['Normal', 'Pneumonia'];
        return tf.tidy(() => {
            const indices = labels.map(label => classNames.indexOf(label));
            return tf.oneHot(indices, classNames.length);
        });
    }

    /**
     * Convert tensors to batch for model input
     */
    createBatch(tensors) {
        return tf.tidy(() => {
            return tf.stack(tensors);
        });
    }

    /**
     * Create training dataset with proper batching
     */
    createTrainingDataset(tensors, labels) {
        const oneHotLabels = this.labelsToOneHot(labels);
        const batchedTensors = this.createBatch(tensors);
        
        return {
            xs: batchedTensors,
            ys: oneHotLabels
        };
    }

    /**
     * Get training data statistics
     */
    getTrainingStats() {
        return {
            normal: this.trainingData.normal.length,
            pneumonia: this.trainingData.pneumonia.length,
            total: this.trainingData.normal.length + this.trainingData.pneumonia.length
        };
    }

    /**
     * Get class distribution for balancing
     */
    getClassDistribution() {
        const stats = this.getTrainingStats();
        const total = stats.total || 1; // Avoid division by zero
        return {
            normal: stats.normal,
            pneumonia: stats.pneumonia,
            total: stats.total,
            normalRatio: (stats.normal / total * 100).toFixed(1),
            pneumoniaRatio: (stats.pneumonia / total * 100).toFixed(1)
        };
    }

    /**
     * Utility function to shuffle arrays
     */
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    /**
     * Clear all loaded data and free memory
     */
    clearPreviousData() {
        // Dispose all tensors to free GPU memory
        this.processedTensors.forEach(tensor => {
            if (tensor && !tensor.isDisposed) {
                tensor.dispose();
            }
        });
        
        this.loadedImages = [];
        this.processedTensors = [];
        this.labels = [];
        this.trainingData = {
            normal: [],
            pneumonia: []
        };
        
        // Force garbage collection
        if (tf.memory().numTensors > 0) {
            console.warn(`Memory leak detected: ${tf.memory().numTensors} tensors remaining`);
            // Try to clean up
            tf.engine().startScope();
            tf.engine().endScope();
        }
    }

    /**
     * Get memory usage statistics
     */
    getMemoryStats() {
        return tf.memory();
    }

    /**
     * Validate medical image quality (basic checks)
     */
    validateImageQuality(imageData) {
        const issues = [];
        
        if (imageData.originalWidth < 512 || imageData.originalHeight < 512) {
            issues.push('Low resolution - medical images typically require higher resolution');
        }
        
        // Check aspect ratio (chest X-rays are typically portrait)
        const aspectRatio = imageData.originalWidth / imageData.originalHeight;
        if (aspectRatio > 1.5 || aspectRatio < 0.67) {
            issues.push('Unusual aspect ratio for chest X-ray');
        }
        
        return issues;
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MedicalImageLoader;
}
