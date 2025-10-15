import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class DataLoader {
    constructor() {
        this.data = null;
        this.symbols = [];
        this.dates = [];
        this.normalizedData = null;
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Parse data rows
        const rawData = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, idx) => {
                row[header] = values[idx];
            });
            rawData.push(row);
        }

        // Extract unique symbols and dates
        this.symbols = [...new Set(rawData.map(row => row.Symbol))].sort();
        this.dates = [...new Set(rawData.map(row => row.Date))].sort();
        
        // Pivot data: dates × symbols × features
        this.data = {};
        this.symbols.forEach(symbol => {
            this.data[symbol] = {};
            this.dates.forEach(date => {
                const row = rawData.find(r => r.Date === date && r.Symbol === symbol);
                if (row) {
                    this.data[symbol][date] = {
                        Open: parseFloat(row.Open),
                        High: parseFloat(row.High),
                        Low: parseFloat(row.Low),
                        Close: parseFloat(row.Close)
                    };
                }
            });
        });
    }

    normalizeData() {
        this.normalizedData = {};
        
        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            const prices = Object.values(this.data[symbol]).flatMap(day => 
                [day.Open, day.High, day.Low, day.Close]
            );
            const min = Math.min(...prices);
            const max = Math.max(...prices);
            
            this.dates.forEach(date => {
                if (this.data[symbol][date]) {
                    this.normalizedData[symbol][date] = {
                        Open: (this.data[symbol][date].Open - min) / (max - min),
                        High: (this.data[symbol][date].High - min) / (max - min),
                        Low: (this.data[symbol][date].Low - min) / (max - min),
                        Close: (this.data[symbol][date].Close - min) / (max - min)
                    };
                }
            });
        });
    }

    createSamples(sequenceLength = 30, predictionHorizon = 3) {
        if (!this.normalizedData) this.normalizeData();
        
        const samples = [];
        const labels = [];
        
        // Use first 80% for training, last 20% for testing
        const splitIndex = Math.floor(this.dates.length * 0.8);
        const trainDates = this.dates.slice(0, splitIndex);
        const testDates = this.dates.slice(splitIndex);
        
        // Create samples for training and testing
        this.createSamplesForDates(trainDates, samples, labels, sequenceLength, predictionHorizon);
        this.createSamplesForDates(testDates, samples, labels, sequenceLength, predictionHorizon);
        
        // Convert to tensors
        const X = tf.tensor3d(samples);
        const y = tf.tensor2d(labels);
        
        // Split back into train/test
        const trainSize = trainDates.length - sequenceLength - predictionHorizon + 1;
        this.X_train = X.slice([0, 0, 0], [trainSize, sequenceLength, 40]);
        this.y_train = y.slice([0, 0], [trainSize, 30]);
        this.X_test = X.slice([trainSize, 0, 0], [samples.length - trainSize, sequenceLength, 40]);
        this.y_test = y.slice([trainSize, 0], [samples.length - trainSize, 30]);
        
        // Clean up intermediate tensors
        X.dispose();
        y.dispose();
    }

    createSamplesForDates(dateSubset, samples, labels, sequenceLength, predictionHorizon) {
        for (let i = 0; i <= dateSubset.length - sequenceLength - predictionHorizon; i++) {
            const sequence = [];
            
            // Create input sequence (30 days × 40 features)
            for (let j = 0; j < sequenceLength; j++) {
                const date = dateSubset[i + j];
                const features = [];
                
                this.symbols.forEach(symbol => {
                    if (this.normalizedData[symbol][date]) {
                        features.push(
                            this.normalizedData[symbol][date].Open,
                            this.normalizedData[symbol][date].High,
                            this.normalizedData[symbol][date].Low,
                            this.normalizedData[symbol][date].Close
                        );
                    } else {
                        // Handle missing data with zeros
                        features.push(0, 0, 0, 0);
                    }
                });
                sequence.push(features);
            }
            samples.push(sequence);

            // Create labels (10 stocks × 3 days = 30 binary outputs)
            const labelRow = [];
            const baseDate = dateSubset[i + sequenceLength - 1];
            const baseCloses = this.symbols.map(symbol => 
                this.data[symbol][baseDate]?.Close || 0
            );

            for (let offset = 1; offset <= predictionHorizon; offset++) {
                const futureDate = dateSubset[i + sequenceLength - 1 + offset];
                this.symbols.forEach((symbol, idx) => {
                    const futureClose = this.data[symbol][futureDate]?.Close;
                    if (futureClose && baseCloses[idx] > 0) {
                        labelRow.push(futureClose > baseCloses[idx] ? 1 : 0);
                    } else {
                        labelRow.push(0); // Default to 0 for missing data
                    }
                });
            }
            labels.push(labelRow);
        }
    }

    getSymbols() {
        return this.symbols;
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}
