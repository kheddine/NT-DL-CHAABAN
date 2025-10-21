// data-loader.js
class DataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.normalizers = {};
        this.encoders = {};
        this.products = [];
        this.featureNames = [];
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
                    this.rawData = results.data.filter(row => row.Date && row.Order_Demand != null);
                    resolve(this.rawData);
                },
                error: (error) => reject(error)
            });
        });
    }

    preprocessData() {
        if (!this.rawData) throw new Error('No data loaded');
        
        // Parse dates and sort chronologically
        this.rawData.forEach(row => {
            row.Date = new Date(row.Date);
            row.Order_Demand = Number(row.Order_Demand);
        });
        
        this.rawData.sort((a, b) => a.Date - b.Date);

        // Handle missing values
        this.rawData.forEach(row => {
            row.Promo = row.Promo || 0;
            row.Open = row.Open !== undefined ? row.Open : 1;
            row.Petrol_price = row.Petrol_price || 0;
            row.StateHoliday = String(row.StateHoliday || '0');
            row.SchoolHoliday = row.SchoolHoliday || 0;
            row.Warehouse = String(row.Warehouse || 'Unknown');
            row.Product_Category = String(row.Product_Category || 'Unknown');
        });

        // Extract unique products
        this.products = [...new Set(this.rawData.map(row => row.Product_id))].filter(Boolean);
        
        this.processedData = this.rawData;
        return this.processedData;
    }

    prepareProductData(productId) {
        if (!this.processedData) throw new Error('No processed data available');
        
        // Filter data for specific product
        const productData = this.processedData.filter(row => row.Product_id === productId);
        
        if (productData.length < 37) {
            throw new Error(`Insufficient data for product ${productId}. Need at least 37 records, got ${productData.length}`);
        }

        // Aggregate by date
        const dailyData = {};
        productData.forEach(row => {
            const dateStr = row.Date.toISOString().split('T')[0];
            if (!dailyData[dateStr]) {
                dailyData[dateStr] = {
                    Date: row.Date,
                    Order_Demand: 0,
                    Open: row.Open,
                    Promo: row.Promo,
                    StateHoliday: row.StateHoliday,
                    SchoolHoliday: row.SchoolHoliday,
                    Petrol_price: row.Petrol_price,
                    Warehouse: row.Warehouse,
                    Product_Category: row.Product_Category,
                    count: 0
                };
            }
            dailyData[dateStr].Order_Demand += row.Order_Demand;
            dailyData[dateStr].count++;
        });

        const aggregatedData = Object.values(dailyData).sort((a, b) => a.Date - b.Date);

        // Prepare features and encoders
        this.prepareEncoders(aggregatedData);
        
        // Create feature matrix
        const features = aggregatedData.map(row => this.encodeRow(row));
        
        // Normalize features
        const normalizedFeatures = this.normalizeFeatures(features);
        
        // Create sequences
        const sequences = this.createSequences(normalizedFeatures);
        
        return {
            sequences: sequences.features,
            labels: sequences.labels,
            dates: aggregatedData.map(row => row.Date),
            originalData: aggregatedData
        };
    }

    prepareEncoders(data) {
        // One-hot encode categorical variables
        const categoricalColumns = ['Warehouse', 'Product_Category', 'StateHoliday'];
        
        categoricalColumns.forEach(col => {
            if (!this.encoders[col]) {
                const uniqueValues = [...new Set(data.map(row => row[col]))];
                this.encoders[col] = {};
                uniqueValues.forEach((value, index) => {
                    this.encoders[col][value] = index;
                });
            }
        });

        // Define feature names for debugging
        this.featureNames = [
            'Order_Demand', 'Open', 'Promo', 'SchoolHoliday', 'Petrol_price',
            ...categoricalColumns.flatMap(col => 
                Object.keys(this.encoders[col]).map(val => `${col}_${val}`)
            )
        ];
    }

    encodeRow(row) {
        const numericalFeatures = [
            row.Order_Demand,
            row.Open,
            row.Promo,
            row.SchoolHoliday,
            row.Petrol_price
        ];

        const categoricalFeatures = [];
        
        // One-hot encode categorical variables
        ['Warehouse', 'Product_Category', 'StateHoliday'].forEach(col => {
            const encoder = this.encoders[col];
            const encoded = new Array(Object.keys(encoder).length).fill(0);
            if (encoder[row[col]] !== undefined) {
                encoded[encoder[row[col]]] = 1;
            }
            categoricalFeatures.push(...encoded);
        });

        return [...numericalFeatures, ...categoricalFeatures];
    }

    normalizeFeatures(features) {
        const numFeatures = features[0].length;
        const normalized = [];
        
        for (let i = 0; i < numFeatures; i++) {
            const column = features.map(row => row[i]);
            
            if (!this.normalizers[i]) {
                this.normalizers[i] = {
                    min: Math.min(...column),
                    max: Math.max(...column)
                };
            }
            
            const norm = this.normalizers[i];
            normalized.push(column.map(val => 
                (val - norm.min) / (norm.max - norm.min || 1)
            ));
        }
        
        // Transpose back to row format
        return normalized[0].map((_, colIndex) => 
            normalized.map(row => row[colIndex])
        );
    }

    createSequences(normalizedData) {
        const sequenceLength = 30;
        const forecastDays = 7;
        const features = [];
        const labels = [];

        for (let i = 0; i < normalizedData.length - sequenceLength - forecastDays + 1; i++) {
            const featureWindow = normalizedData.slice(i, i + sequenceLength);
            const labelWindow = normalizedData
                .slice(i + sequenceLength, i + sequenceLength + forecastDays)
                .map(row => row[0]); // Order_Demand is first feature

            features.push(featureWindow);
            labels.push(labelWindow);
        }

        return {
            features: tf.tensor3d(features),
            labels: tf.tensor2d(labels)
        };
    }

    splitData(features, labels, splitRatio = 0.8) {
        const splitIndex = Math.floor(features.shape[0] * splitRatio);
        
        const X_train = features.slice([0, 0, 0], [splitIndex, -1, -1]);
        const X_test = features.slice([splitIndex, 0, 0], [-1, -1, -1]);
        const y_train = labels.slice([0, 0], [splitIndex, -1]);
        const y_test = labels.slice([splitIndex, 0], [-1, -1]);

        return { X_train, X_test, y_train, y_test };
    }

    denormalizePredictions(normalizedPredictions) {
        if (!this.normalizers[0]) throw new Error('Normalizers not initialized');
        
        const norm = this.normalizers[0]; // Order_Demand normalizer
        return normalizedPredictions.map(pred => 
            pred * (norm.max - norm.min) + norm.min
        );
    }

    dispose() {
        if (this.features) this.features.dispose();
        if (this.labels) this.labels.dispose();
    }

    getProducts() {
        return this.products;
    }
}

export default DataLoader;
