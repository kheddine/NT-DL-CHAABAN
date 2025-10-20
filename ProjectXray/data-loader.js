/**
 * Medical Image Data Loader and Preprocessor
 * Handles chest X-ray image loading from folders, preprocessing, and augmentation
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
        this.augmentationOptions = {
            rotation: true,
            flipping: true,
            brightness: true
        };
    }

    /**
     * Load training data from separate normal and pneumonia folders
     */
    async loadTrainingData(normalFiles, pneumoniaFiles) {
        this.clearPreviousData();
        
        console.log(`Loading training data: ${normalFiles.length} normal, ${pneumoniaFiles.length} pneumonia images`);
        
        // Load normal images
        for (const file of normalFiles) {
            try {
                const imageData = await this.loadSingleImage(file);
                this.trainingData.normal.push(imageData);
            } catch (error) {
                console.warn(`Error loading normal image ${file.name}:`, error);
            }
        }
        
        // Load pneumonia images
        for (const file of pneumoniaFiles) {
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
     * Load test data from folder or individual files
     */
    async loadTestData(files) {
        this.clearPreviousData();
        
        const validFiles = this.validateFiles(files);
        if (validFiles.length > 50) {
            throw new Error('Maximum 50 images allowed. Please select fewer files.');
        }

        // Load images sequentially to avoid memory issues
        for (const file of validFiles) {
            try {
                const imageData = await this.loadSingleImage(file);
                this.loadedImages.push(imageData);
            } catch (error) {
                console.error(`Error loading image ${file.name}:`, error);
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
     * Prepare training dataset from loaded folders
     */
    prepareTrainingDataset() {
        if (this.trainingData.normal.length === 0 && this.trainingData.pneumonia.length === 0) {
            throw new Error('No training data loaded. Please load normal and pneumonia folders first.');
        }

        const allTensors = [];
        const allLabels = [];
        
        // Process normal images
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
    preprocessImages(images, targetSize = [224, 224]) {
        this.processedTensors = [];
        this.labels = []; // Clear labels for inference
        
        images.forEach((imageData, index) => {
            try {
                const tensor = this.preprocessSingleImage(imageData.element, targetSize);
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
    preprocessSingleImage(imgElement, targetSize = [224, 224]) {
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
            
            // Resize to target size (224x224 for most CNN architectures)
            tensor = tf.image.resizeBilinear(tensor, targetSize);
            
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
            return tensor.sub(min).div(max.sub(min));
        });
    }

    /**
     * Data augmentation for training medical imaging models
     */
    augmentData(tensors, labels, augmentationCount = 2) {
        const augmentedTensors = [];
        const augmentedLabels = [];
        
        tensors.forEach((tensor, index) => {
            // Add original tensor
            augmentedTensors.push(tensor);
            augmentedLabels.push(labels[index]);
            
            // Create augmented versions
            for (let i = 0; i < augmentationCount; i++) {
                const augmentedTensor = this.applyAugmentation(tensor);
                augmentedTensors.push(augmentedTensor);
                augmentedLabels.push(labels[index]);
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
            
            // Random rotation (-15 to +15 degrees) - limited for medical images
            if (this.augmentationOptions.rotation && Math.random() > 0.5) {
                const rotationAngle = (Math.random() - 0.5) * 30; // ±15 degrees
                augmented = this.rotateTensor(augmented, rotationAngle * Math.PI / 180);
            }
            
            // Random horizontal flip - common in medical imaging
            if (this.augmentationOptions.flipping && Math.random() > 0.5) {
                augmented = augmented.reverse(1); // Horizontal flip
            }
            
            // Random brightness adjustment
            if (this.augmentationOptions.brightness && Math.random() > 0.5) {
                const brightnessDelta = (Math.random() - 0.5) * 0.2; // ±10%
                augmented = tf.clipByValue(augmented.add(brightnessDelta), 0, 1);
            }
            
            return augmented;
        });
    }

    /**
     * Rotate tensor for data augmentation
     */
    rotateTensor(tensor, angle) {
        return tf.tidy(() => {
            const [height, width] = tensor.shape;
            const centerX = width / 2;
            const centerY = height / 2;
            
            // Create rotation matrix
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            
            // For simplicity, using affine transform
            // In production, consider using tf.image.transform with proper rotation
            return tensor; // Placeholder - implement proper rotation if needed
        });
    }

    /**
     * Split data into training and testing sets
     */
    splitData(tensors, labels, testSplit = 0.2) {
        // Separate indices by class for stratified split
        const normalIndices = labels.map((label, idx) => label === 'Normal' ? idx : -1).filter(idx => idx !== -1);
        const pneumoniaIndices = labels.map((label, idx) => label === 'Pneumonia' ? idx : -1).filter(idx => idx !== -1);
        
        // Shuffle each class
        this.shuffleArray(normalIndices);
        this.shuffleArray(pneumoniaIndices);
        
        // Calculate test size for each class
        const normalTestSize = Math.floor(normalIndices.length * testSplit);
        const pneumoniaTestSize = Math.floor(pneumoniaIndices.length * testSplit);
        
        // Create test indices
        const testIndices = [
            ...normalIndices.slice(0, normalTestSize),
            ...pneumoniaIndices.slice(0, pneumoniaTestSize)
        ];
        
        // Create train indices
        const trainIndices = [
            ...normalIndices.slice(normalTestSize),
            ...pneumoniaIndices.slice(pneumoniaTestSize)
        ];
        
        // Shuffle final arrays
        this.shuffleArray(testIndices);
        this.shuffleArray(trainIndices);
        
        const trainTensors = trainIndices.map(i => tensors[i]);
        const trainLabels = trainIndices.map(i => labels[i]);
        const testTensors = testIndices.map(i => tensors[i]);
        const testLabels = testIndices.map(i => labels[i]);
        
        console.log(`Data split: ${trainTensors.length} training, ${testTensors.length} test images`);
        
        return {
            train: { tensors: trainTensors, labels: trainLabels },
            test: { tensors: testTensors, labels: testLabels }
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
    createTrainingDataset(tensors, labels, batchSize = 8) {
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
        return {
            normal: stats.normal,
            pneumonia: stats.pneumonia,
            total: stats.total,
            normalRatio: (stats.normal / stats.total * 100).toFixed(1),
            pneumoniaRatio: (stats.pneumonia / stats.total * 100).toFixed(1)
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
        
        // Force garbage collection if available
        if (tf.memory().numTensors > 0) {
            console.warn(`Memory leak detected: ${tf.memory().numTensors} tensors remaining`);
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
