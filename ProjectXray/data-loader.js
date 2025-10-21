/**
 * Medical Image Data Loader and Preprocessor
 * Handles chest X-ray image loading, preprocessing, and augmentation
 * FOR EDUCATIONAL PURPOSES ONLY
 */

class MedicalImageLoader {
    constructor() {
        this.processedImages = [];
        this.trainingData = null;
        this.validationData = null;
        this.labelMap = {
            'covid': 1,
            'normal': 0,
            'COVID-19': 1,
            'Normal': 0
        };
    }

    /**
     * Load and preprocess single image for medical analysis
     */
    async loadImage(file) {
        return new Promise((resolve, reject) => {
            // Validate file type and size
            if (!this.validateImageFile(file)) {
                reject(new Error('Invalid image file'));
                return;
            }

            const reader = new FileReader();
            
            reader.onload = async (event) => {
                try {
                    const img = new Image();
                    img.onload = async () => {
                        try {
                            const processedTensor = await this.preprocessImage(img, file.name);
                            resolve({
                                tensor: processedTensor,
                                originalFile: file,
                                originalImage: img,
                                filename: file.name
                            });
                        } catch (error) {
                            reject(error);
                        }
                    };
                    img.onerror = () => reject(new Error('Failed to load image'));
                    img.src = event.target.result;
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    /**
     * Validate medical image file
     */
    validateImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!validTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please use JPG or PNG images.');
        }

        if (file.size > maxSize) {
            throw new Error('File too large. Maximum size is 10MB.');
        }

        return true;
    }

    /**
     * Preprocess chest X-ray image for CNN input
     * Medical imaging-specific preprocessing
     */
    async preprocessImage(img, filename) {
        return tf.tidy(() => {
            // Convert to tensor
            let tensor = tf.browser.fromPixels(img);
            
            // Medical imaging: Handle grayscale conversion if needed
            if (tensor.shape[2] === 1) {
                // Already grayscale - convert to RGB
                tensor = tensor.stack().transpose([1, 2, 0]);
            } else if (tensor.shape[2] === 4) {
                // Remove alpha channel
                tensor = tensor.slice([0, 0, 0], [tensor.shape[0], tensor.shape[1], 3]);
            }

            // Resize to model input size (224x224 for medical CNNs)
            const resized = tf.image.resizeBilinear(tensor, [224, 224]);
            
            // Normalize pixel values to [0, 1] range
            const normalized = resized.div(255.0);
            
            // Medical imaging: Apply contrast enhancement for X-rays
            const enhanced = this.enhanceContrast(normalized);
            
            return enhanced;
        });
    }

    /**
     * Contrast enhancement for chest X-ray images
     */
    enhanceContrast(tensor) {
        return tf.tidy(() => {
            // Simple contrast stretching for medical images
            const min = tensor.min();
            const max = tensor.max();
            const stretched = tensor.sub(min).div(max.sub(min));
            
            // Gamma correction for better visualization
            const gamma = 1.2; // Adjust based on X-ray characteristics
            const corrected = stretched.pow(gamma);
            
            return corrected;
        });
    }

    /**
     * Load multiple images for batch processing
     */
    async loadMultipleImages(files) {
        const results = [];
        const errors = [];

        for (const file of files) {
            try {
                const processed = await this.loadImage(file);
                results.push(processed);
            } catch (error) {
                errors.push({
                    file: file.name,
                    error: error.message
                });
                console.warn(`Failed to process ${file.name}:`, error);
            }
        }

        return { results, errors };
    }

    /**
     * Extract label from filename for medical images
     * Common patterns in medical datasets
     */
    extractLabel(filename) {
        const lowerName = filename.toLowerCase();
        
        // Common COVID-19 dataset naming patterns
        if (lowerName.includes('covid') || 
            lowerName.includes('cov') || 
            lowerName.includes('positive')) {
            return 'COVID-19';
        }
        
        if (lowerName.includes('normal') || 
            lowerName.includes('healthy') || 
            lowerName.includes('negative')) {
            return 'Normal';
        }

        // Try to infer from folder structure in directory uploads
        const pathParts = filename.split('/');
        if (pathParts.length > 1) {
            const folderName = pathParts[pathParts.length - 2].toLowerCase();
            if (this.labelMap.hasOwnProperty(folderName)) {
                return folderName === 'covid' ? 'COVID-19' : 'Normal';
            }
        }

        // Default to normal if uncertain (conservative approach)
        console.warn(`Unable to determine label for ${filename}, defaulting to Normal`);
        return 'Normal';
    }

    /**
     * Prepare training dataset with medical data considerations
     */
    prepareTrainingDataset(images, testSplit = 0.2) {
        if (images.length === 0) {
            throw new Error('No images available for training');
        }

        return tf.tidy(() => {
            // Extract features and labels
            const features = [];
            const labels = [];
            const labelIndices = [];

            images.forEach(imgData => {
                features.push(imgData.tensor);
                const label = this.extractLabel(imgData.filename);
                labels.push(label);
                labelIndices.push(this.labelMap[label]);
            });

            // Convert to tensors
            const featuresTensor = tf.stack(features);
            const labelsTensor = tf.oneHot(tf.tensor1d(labelIndices, 'int32'), 2);

            // Split into training and validation sets
            const splitIndex = Math.floor(featuresTensor.shape[0] * (1 - testSplit));
            
            const trainFeatures = featuresTensor.slice(0, splitIndex);
            const trainLabels = labelsTensor.slice(0, splitIndex);
            
            const valFeatures = featuresTensor.slice(splitIndex);
            const valLabels = labelsTensor.slice(splitIndex);

            this.trainingData = {
                xs: trainFeatures,
                ys: trainLabels,
                labels: labels.slice(0, splitIndex)
            };

            this.validationData = {
                xs: valFeatures,
                ys: valLabels,
                labels: labels.slice(splitIndex)
            };

            console.log(`Training set: ${trainFeatures.shape[0]} images`);
            console.log(`Validation set: ${valFeatures.shape[0]} images`);

            return {
                training: this.trainingData,
                validation: this.validationData,
                classDistribution: this.calculateClassDistribution(labels)
            };
        });
    }

    /**
     * Calculate class distribution for medical dataset
     */
    calculateClassDistribution(labels) {
        const distribution = {
            'COVID-19': 0,
            'Normal': 0
        };

        labels.forEach(label => {
            distribution[label]++;
        });

        return distribution;
    }

    /**
     * Data augmentation for medical images
     * Preserves medical integrity while increasing diversity
     */
    augmentImage(tensor) {
        return tf.tidy(() => {
            const augmentations = [];
            
            // Random rotation (small angles for medical images)
            const rotation = tf.rotateWithOffset(tensor, Math.random() * 0.1);
            augmentations.push(rotation);
            
            // Random flip (horizontal only for chest X-rays)
            if (Math.random() > 0.5) {
                const flipped = tf.reverse(tensor, 1);
                augmentations.push(flipped);
            }
            
            // Random brightness adjustment (conservative for medical)
            const brightness = tf.add(tensor, tf.randomUniform([], -0.1, 0.1));
            augmentations.push(brightness);
            
            // Choose one random augmentation
            const randomAug = augmentations[Math.floor(Math.random() * augmentations.length)];
            
            return randomAug.clipByValue(0, 1);
        });
    }

    /**
     * Create augmented batch for training
     */
    createAugmentedBatch(originalBatch, augmentationFactor = 2) {
        return tf.tidy(() => {
            const augmentedFeatures = [];
            const augmentedLabels = [];

            for (let i = 0; i < originalBatch.xs.shape[0]; i++) {
                const originalFeature = originalBatch.xs.slice([i, 0, 0, 0], [1, ...originalBatch.xs.shape.slice(1)]);
                const originalLabel = originalBatch.ys.slice([i, 0], [1, 2]);
                
                // Add original
                augmentedFeatures.push(originalFeature);
                augmentedLabels.push(originalLabel);
                
                // Add augmented versions
                for (let j = 0; j < augmentationFactor; j++) {
                    const augmented = this.augmentImage(originalFeature.squeeze());
                    augmentedFeatures.push(augmented.expandDims(0));
                    augmentedLabels.push(originalLabel);
                }
            }

            return {
                xs: tf.concat(augmentedFeatures),
                ys: tf.concat(augmentedLabels)
            };
        });
    }

    /**
     * Clear memory and reset state
     */
    dispose() {
        // Dispose all tensors
        if (this.trainingData) {
            this.trainingData.xs.dispose();
            this.trainingData.ys.dispose();
        }
        
        if (this.validationData) {
            this.validationData.xs.dispose();
            this.validationData.ys.dispose();
        }
        
        this.processedImages.forEach(img => {
            if (img.tensor) {
                img.tensor.dispose();
            }
        });
        
        this.processedImages = [];
        this.trainingData = null;
        this.validationData = null;
        
        // Force garbage collection if available
        if (tf.memory().numTensors > 0) {
            console.warn(`Memory leak detected: ${tf.memory().numTensors} tensors remaining`);
            tf.disposeVariables();
        }
    }

    /**
     * Get memory usage statistics
     */
    getMemoryStats() {
        return tf.memory();
    }
}

// Initialize global data loader
const medicalDataLoader = new MedicalImageLoader();
