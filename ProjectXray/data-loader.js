// data-loader.js
class DataLoader {
    constructor() {
        this.data = null;
        this.products = [];
    }

    loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error('CSV parsing failed: ' + results.errors[0].message));
                        return;
                    }
                    this.data = results.data;
                    resolve(this.data);
                },
                error: (error) => reject(error)
            });
        });
    }

    preprocessData() {
        if (!this.data) throw new Error('No data loaded');
        
        // Basic preprocessing
        this.data = this.data.filter(row => 
            row.Date && row.Order_Demand !== undefined && row.Product_id
        );
        
        // Parse dates and convert demand to numbers
        this.data.forEach(row => {
            row.Date = new Date(row.Date);
            row.Order_Demand = Number(row.Order_Demand);
        });
        
        // Get unique products
        this.products = [...new Set(this.data.map(row => row.Product_id))].filter(Boolean);
        
        return this.data;
    }

    prepareProductData(productId) {
        const productData = this.data.filter(row => row.Product_id === productId);
        
        if (productData.length < 37) {
            throw new Error(`Not enough data for product ${productId}. Need at least 37 records, got ${productData.length}`);
        }
        
        // Simple feature preparation (just using demand for now)
        const demands = productData.map(row => row.Order_Demand);
        
        // Normalize demands
        const min = Math.min(...demands);
        const max = Math.max(...demands);
        const normalized = demands.map(d => (d - min) / (max - min));
        
        // Create sequences
        const sequenceLength = 30;
        const forecastDays = 7;
        const features = [];
        const labels = [];
        
        for (let i = 0; i < normalized.length - sequenceLength - forecastDays + 1; i++) {
            features.push(normalized.slice(i, i + sequenceLength));
            labels.push(normalized.slice(i + sequenceLength, i + sequenceLength + forecastDays));
        }
        
        return {
            sequences: tf.tensor3d(features.map(f => f.map(val => [val]))), // Shape: [samples, 30, 1]
            labels: tf.tensor2d(labels),
            normalizer: { min, max }
        };
    }

    splitData(features, labels, splitRatio = 0.8) {
        const splitIndex = Math.floor(features.shape[0] * splitRatio);
        
        return {
            X_train: features.slice([0, 0, 0], [splitIndex, -1, -1]),
            X_test: features.slice([splitIndex, 0, 0], [-1, -1, -1]),
            y_train: labels.slice([0, 0], [splitIndex, -1]),
            y_test: labels.slice([splitIndex, 0], [-1, -1])
        };
    }

    getProducts() {
        return this.products;
    }
}

export default DataLoader;
