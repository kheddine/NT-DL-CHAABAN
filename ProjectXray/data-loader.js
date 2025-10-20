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
        this.augmentationOptions = {
            rotation: true,
            flipping: true,
            brightness: true
        };
    }

    /**
     * Load and validate medical images from file input
     */
    async loadImagesFromFiles(files) {
        this.clearPreviousData();
        
        const validFiles = Array.from(files).filter(file => {
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

        console.log(`Successfully loaded ${this.loadedImages.length} images`);
        return this.loadedImages;
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
     * Preprocess images for CNN model input
     * Medical imaging specific preprocessing
     */
    preprocessImages(images, targetSize = [224, 224]) {
        this.processedTensors = [];
        
        images.forEach((imageData, index) => {
            try {
                const tensor = this.preprocessSingleImage(imageData.element, targetSize);
                this.processedTensors.push(tensor);
                
                // Assign placeholder labels for demonstration
                // In real scenario, these would come from user input or metadata
                this.labels.push(index % 2 === 0 ? 'COVID-19' : 'Normal');
                
            } catch (error) {
                console.error(`Error preprocessing image ${index}:`, error);
            }
        });
        
        console.log(`Preprocessed ${this.processedTensors.length} images`);
        return {
            tensors: this.processedTensors,
            labels: this.labels
        };
    }

    /**
     * Preprocess single image with medical imaging considerations
     */
    preprocessSingleImage(imgElement, targetSize) {
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
        const indices = Array.from({length: tensors.length}, (_, i) => i);
        this.shuffleArray(indices);
        
        const testSize = Math.floor(tensors.length * testSplit);
        const testIndices = indices.slice(0, testSize);
        const trainIndices = indices.slice(testSize);
        
        const trainTensors = trainIndices.map(i => tensors[i]);
        const trainLabels = trainIndices.map(i => labels[i]);
        const testTensors = testIndices.map(i => tensors[i]);
        const testLabels = testIndices.map(i => labels[i]);
        
        return {
            train: { tensors: trainTensors, labels: trainLabels },
            test: { tensors: testTensors, labels: testLabels }
        };
    }

    /**
     * Convert labels to one-hot encoding for categorical classification
     */
    labelsToOneHot(labels) {
        const classNames = ['Normal', 'COVID-19'];
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
     * Generate synthetic medical images for demonstration
     * In real scenario, this would use actual medical data
     */
    generateSyntheticData(sampleCount = 20) {
        const syntheticTensors = [];
        const syntheticLabels = [];
        
        for (let i = 0; i < sampleCount; i++) {
            const tensor = tf.tidy(() => {
                // Create synthetic chest X-ray like patterns
                const base = tf.randomUniform([224, 224, 3], 0.1, 0.3);
                
                // Add lung-like circular patterns
                const lungs = this.createSyntheticLungs();
                
                // Add some random "pathology" for COVID samples
                const hasPathology = i % 2 === 0;
                if (hasPathology) {
                    const pathology = tf.randomUniform([224, 224, 3], 0, 0.2);
                    return tf.clipByValue(base.add(lungs).add(pathology), 0, 1);
                }
                
                return tf.clipByValue(base.add(lungs), 0, 1);
            });
            
            syntheticTensors.push(tensor);
            syntheticLabels.push(i % 2 === 0 ? 'COVID-19' : 'Normal');
        }
        
        return {
            tensors: syntheticTensors,
            labels: syntheticLabels
        };
    }

    /**
     * Create synthetic lung patterns for demonstration
     */
    createSyntheticLungs() {
        return tf.tidy(() => {
            const lungs = tf.zeros([224, 224, 3]);
            // This would create circular patterns resembling lungs
            // Simplified for demonstration
            return tf.randomUniform([224, 224, 3], 0.05, 0.15);
        });
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
