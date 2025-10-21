class DataLoader {
    constructor() {
        this.rawData = [];
        this.processedData = {};
        this.featureInfo = {
            min: {},
            max: {},
            categories: {}
        };
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(`CSV parsing errors: ${results.errors.map(e => e.message).join(', ')}`));
                        return;
                    }
                    this.rawData = results.data.filter(row => 
                        row.Product_id && 
                        row.Date && 
                        !isNaN(row.Order_Demand) &&
                        row.Order_Demand > 0
                    );
                    resolve(this.rawData);
                },
                error: (error) => reject(error)
            });
        });
    }

    preprocessData() {
        // Group by product
        const products = {};
        this.rawData.forEach(row => {
            if (!products[row.Product_id]) {
                products[row.Product_id] = [];
            }
            products[row.Product_id].push({
                ...row,
                Date: new Date(row.Date)
            });
        });

        // Sort each product by date and process
        Object.keys(products).forEach(productId => {
            products[productId].sort((a, b) => a.Date - b.Date);
            this.processProductData(productId, products[productId]);
        });

        return this.processedData;
    }

    processProductData(productId, productData) {
        if (productData.length < 37) return; // Need at least 37 days for 30-day input + 7-day output

        // Extract and normalize features
        const sequences = [];
        const targets = [];
        
        // Collect all values for normalization
        const allDemands = productData.map(d => d.Order_Demand);
        const allPetrol = productData.map(d => d.Petrol_price || 0);
        
        // Initialize feature info if not exists
        if (!this.featureInfo.min.Order_Demand) {
            this.featureInfo.min.Order_Demand = Math.min(...allDemands);
            this.featureInfo.max.Order_Demand = Math.max(...allDemands);
            this.featureInfo.min.Petrol_price = Math.min(...allPetrol);
            this.featureInfo.max.Petrol_price = Math.max(...allPetrol);
            
            // Collect categories
            this.featureInfo.categories.Warehouse = [...new Set(this.rawData.map(d => d.Warehouse))];
            this.featureInfo.categories.Product_Category = [...new Set(this.rawData.map(d => d.Product_Category))];
        }

        // Create sequences
        for (let i = 0; i <= productData.length - 37; i++) {
            const sequence = [];
            const target = [];
            
            // Input sequence (30 days)
            for (let j = i; j < i + 30; j++) {
                const features = this.extractFeatures(productData[j]);
                sequence.push(features);
            }
            
            // Output sequence (next 7 days)
            for (let j = i + 30; j < i + 37; j++) {
                target.push(this.normalizeValue(productData[j].Order_Demand, 'Order_Demand'));
            }
            
            sequences.push(sequence);
            targets.push(target);
        }

        if (sequences.length > 0) {
            this.processedData[productId] = {
                sequences: sequences,
                targets: targets,
                productInfo: {
                    code: productData[0].Product_Code,
                    category: productData[0].Product_Category,
                    warehouse: productData[0].Warehouse
                },
                dates: productData.map(d => d.Date)
            };
        }
    }

    extractFeatures(row) {
        const features = [];
        
        // Normalized Order Demand
        features.push(this.normalizeValue(row.Order_Demand, 'Order_Demand'));
        
        // One-hot encode Warehouse
        const warehouseEncoding = new Array(this.featureInfo.categories.Warehouse.length).fill(0);
        const warehouseIndex = this.featureInfo.categories.Warehouse.indexOf(row.Warehouse);
        if (warehouseIndex !== -1) warehouseEncoding[warehouseIndex] = 1;
        features.push(...warehouseEncoding);
        
        // One-hot encode Product Category
        const categoryEncoding = new Array(this.featureInfo.categories.Product_Category.length).fill(0);
        const categoryIndex = this.featureInfo.categories.Product_Category.indexOf(row.Product_Category);
        if (categoryIndex !== -1) categoryEncoding[categoryIndex] = 1;
        features.push(...categoryEncoding);
        
        // Binary features
        features.push(row.Open || 0);
        features.push(row.Promo || 0);
        features.push(row.StateHoliday || 0);
        features.push(row.SchoolHoliday || 0);
        
        // Normalized Petrol Price
        features.push(this.normalizeValue(row.Petrol_price || 0, 'Petrol_price'));
        
        // Day of week (sine/cosine encoding)
        const dayOfWeek = new Date(row.Date).getDay();
        features.push(Math.sin(2 * Math.PI * dayOfWeek / 7));
        features.push(Math.cos(2 * Math.PI * dayOfWeek / 7));
        
        return features;
    }

    normalizeValue(value, featureName) {
        const min = this.featureInfo.min[featureName];
        const max = this.featureInfo.max[featureName];
        return (value - min) / (max - min);
    }

    denormalizeValue(normalizedValue, featureName) {
        const min = this.featureInfo.min[featureName];
        const max = this.featureInfo.max[featureName];
        return normalizedValue * (max - min) + min;
    }

    prepareTrainTestSplit(testRatio = 0.2) {
        const allSequences = [];
        const allTargets = [];
        const productMap = [];

        Object.keys(this.processedData).forEach(productId => {
            const product = this.processedData[productId];
            for (let i = 0; i < product.sequences.length; i++) {
                allSequences.push(product.sequences[i]);
                allTargets.push(product.targets[i]);
                productMap.push(productId);
            }
        });

        // Split chronologically
        const splitIndex = Math.floor(allSequences.length * (1 - testRatio));
        
        const trainData = {
            sequences: allSequences.slice(0, splitIndex),
            targets: allTargets.slice(0, splitIndex),
            productMap: productMap.slice(0, splitIndex)
        };
        
        const testData = {
            sequences: allSequences.slice(splitIndex),
            targets: allTargets.slice(splitIndex),
            productMap: productMap.slice(splitIndex)
        };

        return { trainData, testData };
    }

    getProductData(productId) {
        return this.processedData[productId];
    }

    getAllProducts() {
        return Object.keys(this.processedData);
    }

    getFeatureInfo() {
        return this.featureInfo;
    }
}

export default DataLoader;
