// data-loader.js
class DataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.features = null;
        this.labels = null;
        this.normalizers = {};
        this.encoders = {};
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
                    this.rawData = results.data;
                    resolve(this.rawData);
                },
                error: (error) => reject(error)
            });
        });
    }

    preprocessData() {
        if (!this.rawData) throw new Error('No data loaded');
        
        // Filter out invalid data
        const validData = this.rawData.filter(row => 
            row.Date && row.Order_Demand != null && !isNaN(row.Order_Demand)
        );

        // Parse dates and sort chronologically
        validData.forEach(row => {
            row.Date = new Date(row.Date);
            row.Order_Demand = Number(row.Order_Demand);
        });
        validData.sort((a, b) => a.Date - b.Date);

        // Handle missing values
        validData.forEach(row => {
            row.Promo = row.Promo || 0;
            row.Open = row.Open !== undefined ? row.Open : 1;
            row.Petrol_price = row.Petrol_price || 0;
            row.StateHoliday = row.StateHoliday || '0';
            row.SchoolHoliday = row.SchoolHoliday || 0;
        });

        this.processedData = validData;
        return this.processedData;
    }

    prepareFeaturesForEntity(entityType, entityId) {
        if (!this.processedData) throw new Error('No processed data available');
        
        // Filter data for specific entity (product or warehouse)
        const entityData = this.processedData.filter(row => 
            entityType === 'product' ? row.Product_id === entityId : row.Warehouse === entityId
        );

        if (entityData.length < 37) {
            throw new Error(`Insufficient data for ${entityType} ${entityId}. Need at least 37 records.`);
        }

        // Group by date and aggregate
        const dailyData = {};
        entityData.forEach(row => {
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
                    count: 0
                };
            }
            dailyData[dateStr].Order_Demand += row.Order_Demand;
            dailyData[dateStr].count++;
        });

        const aggregatedData = Object.values(dailyData).sort((a, b) => a.Date - b.Date);

        // Normalize features
        const featureColumns = ['Order_Demand', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Petrol_price'];
        
        featureColumns.forEach(col => {
            if (!this.normalizers[col]) {
                const values = aggregatedData.map(row => row[col]);
                this.normalizers[col] = {
                    min: Math.min(...values),
                    max: Math.max(...values)
                };
            }
        });

        // Encode categorical features
        if (!this.encoders.StateHoliday) {
            const uniqueHolidays = [...new Set(aggregatedData.map(row => row.StateHoliday))];
            this.encoders.StateHoliday = {};
            uniqueHolidays.forEach((holiday, index) => {
                this.encoders.StateHoliday[holiday] = index / Math.max(1, uniqueHolidays.length - 1);
            });
        }

        // Create normalized features array
        const normalizedData = aggregatedData.map(row => {
            const normalizedRow = featureColumns.map(col => {
                if (col === 'StateHoliday') {
                    return this.encoders.StateHoliday[row[col]];
                }
                const norm = this.normalizers[col];
                return (row[col] - norm.min) / (norm.max - norm.min || 1);
            });
            return normalizedRow;
        });

        // Create sliding windows
        const sequenceLength = 30;
        const forecastDays = 7;
        const features = [];
        const labels = [];

        for (let i = 0; i < normalizedData.length - sequenceLength - forecastDays + 1; i++) {
            const featureWindow = normalizedData.slice(i, i + sequenceLength);
            const labelWindow = normalizedData.slice(i + sequenceLength, i + sequenceLength + forecastDays)
                .map(row => row[0]); // Order_Demand is first column

            features.push(featureWindow);
            labels.push(labelWindow);
        }

        this.features = tf.tensor3d(features);
        this.labels = tf.tensor2d(labels);

        return {
            features: this.features,
            labels: this.labels,
            dates: aggregatedData.map(row => row.Date),
            normalizers: this.normalizers,
            encoders: this.encoders
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

    dispose() {
        if (this.features) this.features.dispose();
        if (this.labels) this.labels.dispose();
    }
}

export default DataLoader;
