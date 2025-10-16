import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/+esm';

export class StockDataLoader {
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
        
        // Validate required columns
        const requiredCols = ['Date', 'Symbol', 'Open', 'Close'];
        if (!requiredCols.every(col => headers.includes(col))) {
            throw new Error('CSV must contain Date, Symbol, Open, Close columns');
        }

        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) continue;
            
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            
            // Convert numeric values
            row.Open = parseFloat(row.Open);
            row.Close = parseFloat(row.Close);
            if (isNaN(row.Open) || isNaN(row.Close)) continue;
            
            data.push(row);
        }

        this.data = data;
        this.prepareDataset();
    }

    prepareDataset() {
        // Extract unique symbols and dates
        this.symbols = [...new Set(this.data.map(row => row.Symbol))].sort();
        this.dates = [...new Set(this.data.map(row => row.Date))].sort();
        
        if (this.symbols.length !== 10) {
            console.warn(`Expected 10 stocks, found ${this.symbols.length}`);
        }

        // Pivot data: date × symbol → features
        const pivotedData = [];
        for (const date of this.dates) {
            const dayData = { date };
            for (const symbol of this.symbols) {
                const row = this.data.find(r => r.Date === date && r.Symbol === symbol);
                if (row) {
                    dayData[`${symbol}_Open`] = row.Open;
                    dayData[`${symbol}_Close`] = row.Close;
                }
            }
            // Only include complete days (all 10 stocks present)
            if (Object.keys(dayData).length === 21) { // 1 date + 20 features
                pivotedData.push(dayData);
            }
        }

        this.normalizeData(pivotedData);
        this.createSequences(pivotedData);
    }

    normalizeData(pivotedData) {
        this.normalizedData = [];
        this.minMax = {};

        // Calculate min-max per stock feature
        for (const symbol of this.symbols) {
            const opens = pivotedData.map(d => d[`${symbol}_Open`]);
            const closes = pivotedData.map(d => d[`${symbol}_Close`]);
            
            this.minMax[`${symbol}_Open`] = {
                min: Math.min(...opens),
                max: Math.max(...opens)
            };
            this.minMax[`${symbol}_Close`] = {
                min: Math.min(...closes),
                max: Math.max(...closes)
            };
        }

        // Normalize data
        for (const day of pivotedData) {
            const normalizedDay = { date: day.date };
            for (const symbol of this.symbols) {
                const openKey = `${symbol}_Open`;
                const closeKey = `${symbol}_Close`;
                
                normalizedDay[openKey] = (day[openKey] - this.minMax[openKey].min) / 
                                       (this.minMax[openKey].max - this.minMax[openKey].min);
                normalizedDay[closeKey] = (day[closeKey] - this.minMax[closeKey].min) / 
                                        (this.minMax[closeKey].max - this.minMax[closeKey].min);
            }
            this.normalizedData.push(normalizedDay);
        }
    }

    createSequences(normalizedData) {
        const sequenceLength = 12;
        const predictionHorizon = 3;
        
        const sequences = [];
        const targets = [];

        for (let i = sequenceLength; i < normalizedData.length - predictionHorizon; i++) {
            // Input sequence: last 12 days
            const sequence = [];
            for (let j = i - sequenceLength; j < i; j++) {
                const features = [];
                for (const symbol of this.symbols) {
                    features.push(normalizedData[j][`${symbol}_Open`]);
                    features.push(normalizedData[j][`${symbol}_Close`]);
                }
                sequence.push(features);
            }

            // Target: binary classification for next 3 days for each stock
            const target = [];
            const currentClosePrices = {};
            
            // Get current close prices for reference
            for (const symbol of this.symbols) {
                currentClosePrices[symbol] = normalizedData[i][`${symbol}_Close`];
            }

            // Calculate binary labels for next 3 days
            for (let offset = 1; offset <= predictionHorizon; offset++) {
                const futureDay = normalizedData[i + offset];
                for (const symbol of this.symbols) {
                    const futureClose = futureDay[`${symbol}_Close`];
                    const currentClose = currentClosePrices[symbol];
                    const label = futureClose > currentClose ? 1 : 0;
                    target.push(label);
                }
            }

            sequences.push(sequence);
            targets.push(target);
        }

        this.splitData(sequences, targets);
    }

    splitData(sequences, targets) {
        const splitIndex = Math.floor(sequences.length * 0.8);
        
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));

        console.log(`Training samples: ${this.X_train.shape[0]}`);
        console.log(`Test samples: ${this.X_test.shape[0]}`);
        console.log(`Input shape: [${this.X_train.shape.join(', ')}]`);
        console.log(`Output shape: [${this.y_train.shape.join(', ')}]`);
    }

    getSymbols() {
        return this.symbols;
    }

    getTestDates() {
        const startIndex = Math.floor(this.normalizedData.length * 0.8) + 12;
        return this.dates.slice(startIndex, startIndex + this.X_test.shape[0]);
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}
