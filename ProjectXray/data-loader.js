class DataLoader {
    constructor() {
        this.trainingData = {
            pneumonia: [],
            normal: []
        };
        this.testData = [];
    }

    async loadImage(file, category = null) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    // Preprocess image
                    const tensor = this.preprocessImage(img);
                    resolve({
                        tensor: tensor,
                        file: file,
                        element: img,
                        category: category
                    });
                };
                img.onerror = () => reject(new Error('Failed to load image'));
                img.src = event.target.result;
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    preprocessImage(img) {
        return tf.tidy(() => {
            // Convert to tensor
            let tensor = tf.browser.fromPixels(img);
            
            // Resize to 224x224
            const resized = tf.image.resizeBilinear(tensor, [224, 224]);
            
            // Normalize to [0, 1]
            const normalized = resized.div(255.0);
            
            return normalized;
        });
    }

    async loadTrainingImages(files, category) {
        const results = [];
        
        for (const file of files) {
            try {
                const imageData = await this.loadImage(file, category);
                results.push(imageData);
            } catch (error) {
                console.error(`Error loading image ${file.name}:`, error);
            }
        }
        
        this.trainingData[category] = this.trainingData[category].concat(results);
        return results;
    }

    async loadTestImages(files) {
        const results = [];
        
        for (const file of files) {
            try {
                const imageData = await this.loadImage(file, 'test');
                results.push(imageData);
            } catch (error) {
                console.error(`Error loading test image ${file.name}:`, error);
            }
        }
        
        this.testData = this.testData.concat(results);
        return results;
    }

    prepareTrainingData(validationSplit = 0.2) {
        const allImages = [
            ...this.trainingData.pneumonia.map(img => ({ ...img, label: 1 })), // COVID-19/Pneumonia
            ...this.trainingData.normal.map(img => ({ ...img, label: 0 }))     // Normal
        ];

        if (allImages.length === 0) {
            throw new Error('No training data available');
        }

        // Shuffle data
        this.shuffleArray(allImages);

        // Split into training and validation
        const splitIndex = Math.floor(allImages.length * (1 - validationSplit));
        const trainingImages = allImages.slice(0, splitIndex);
        const validationImages = allImages.slice(splitIndex);

        // Convert to tensors
        const trainFeatures = tf.stack(trainingImages.map(img => img.tensor));
        const trainLabels = tf.oneHot(tf.tensor1d(trainingImages.map(img => img.label), 2), 2);
        
        const valFeatures = tf.stack(validationImages.map(img => img.tensor));
        const valLabels = tf.oneHot(tf.tensor1d(validationImages.map(img => img.label), 2), 2);

        return {
            training: {
                xs: trainFeatures,
                ys: trainLabels,
                size: trainingImages.length
            },
            validation: {
                xs: valFeatures,
                ys: valLabels,
                size: validationImages.length
            },
            summary: {
                total: allImages.length,
                pneumonia: this.trainingData.pneumonia.length,
                normal: this.trainingData.normal.length,
                training: trainingImages.length,
                validation: validationImages.length
            }
        };
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    clearTrainingData() {
        this.trainingData.pneumonia.forEach(img => img.tensor.dispose());
        this.trainingData.normal.forEach(img => img.tensor.dispose());
        this.trainingData = { pneumonia: [], normal: [] };
    }

    clearTestData() {
        this.testData.forEach(img => img.tensor.dispose());
        this.testData = [];
    }

    getStats() {
        return {
            pneumonia: this.trainingData.pneumonia.length,
            normal: this.trainingData.normal.length,
            test: this.testData.length
        };
    }
}
