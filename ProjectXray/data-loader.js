import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

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
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Validate required columns
        const requiredCols = ['Date', 'Symbol', 'Open', 'Close'];
        if (!requiredCols.every(col => headers.includes(col))) {
            throw new Error('CSV missing required columns: Date, Symbol, Open, Close');
        }

        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                let value = values[index];
                // Handle numeric values
                if (header === 'Open' || header === 'Close') {
                    value = parseFloat(value);
                    if (isNaN(value)) {
                        console.warn(`Invalid numeric value in ${header}: ${values[index]}`);
                        value = 0;
                    }
                }
                row[header] = value;
            });
            data.push(row);
        }

        this.data = data;
        this.prepareData();
    }

    prepareData() {
        // Extract unique symbols and dates
        this.symbols = [...new Set(this.data.map(row => row.Symbol))].sort();
        this.dates = [...new Set(this.data.map(row => row.Date))].sort();
        
        if (this.symbols.length === 0) {
            throw new Error('No valid symbols found in CSV');
        }

        console.log(`Processing ${this.symbols.length} stocks with ${this.dates.length} dates`);

        // Pivot data: date × symbol → features
        this.pivotedData = {};
        this.dates.forEach(date => {
            this.pivotedData[date] = {};
            this.symbols.forEach(symbol => {
                const row = this.data.find(r => r.Date === date && r.Symbol === symbol);
                if (row) {
                    this.pivotedData[date][symbol] = {
                        Open: row.Open,
                        Close: row.Close
                    };
                }
            });
        });

        this.normalizeData();
    }

    normalizeData() {
        this.normalizedData = {};
        this.minMax = {};

        // Calculate min-max per stock
        this.symbols.forEach(symbol => {
            const opens = this.dates.map(date => this.pivotedData[date]?.[symbol]?.Open).filter(v => v != null && !isNaN(v));
            const closes = this.dates.map(date => this.pivotedData[date]?.[symbol]?.Close).filter(v => v != null && !isNaN(v));
            
            if (opens.length === 0 || closes.length === 0) {
                throw new Error(`No valid data for stock ${symbol}`);
            }

            this.minMax[symbol] = {
                Open: { min: Math.min(...opens), max: Math.max(...opens) },
                Close: { min: Math.min(...closes), max: Math.max(...closes) }
            };
        });

        // Normalize data
        this.dates.forEach(date => {
            this.normalizedData[date] = {};
            this.symbols.forEach(symbol => {
                const data = this.pivotedData[date][symbol];
                if (data) {
                    const openRange = this.minMax[symbol].Open.max - this.minMax[symbol].Open.min;
                    const closeRange = this.minMax[symbol].Close.max - this.minMax[symbol].Close.min;
                    
                    this.normalizedData[date][symbol] = {
                        Open: openRange === 0 ? 0.5 : (data.Open - this.minMax[symbol].Open.min) / openRange,
                        Close: closeRange === 0 ? 0.5 : (data.Close - this.minMax[symbol].Close.min) / closeRange
                    };
                }
            });
        });
    }

    createSamples(sequenceLength = 12, predictionHorizon = 3) {
        const samples = [];
        const targets = [];
        
        // Ensure we have enough data
        if (this.dates.length <= sequenceLength + predictionHorizon) {
            throw new Error(`Insufficient data: need more than ${sequenceLength + predictionHorizon} days`);
        }

        const validDates = this.dates.slice(sequenceLength, this.dates.length - predictionHorizon);

        console.log(`Creating samples from ${validDates.length} valid dates`);

        validDates.forEach((date, dateIndex) => {
            const dateIdx = this.dates.indexOf(date);
            const inputSequence = [];
            
            // Get 12-day sequence
            for (let i = dateIdx - sequenceLength; i < dateIdx; i++) {
                const seqDate = this.dates[i];
                const features = [];
                
                this.symbols.forEach(symbol => {
                    const stockData = this.normalizedData[seqDate][symbol];
                    if (stockData) {
                        features.push(stockData.Open, stockData.Close);
                    } else {
                        // Fill with zeros if data missing
                        features.push(0, 0);
                    }
                });
                
                inputSequence.push(features);
            }

            // Get target labels for D+1, D+2, D+3
            const targetLabels = [];
            const currentClosePrices = {};
            
            // Get current close prices (date D)
            this.symbols.forEach(symbol => {
                currentClosePrices[symbol] = this.pivotedData[date][symbol].Close;
            });

            // Calculate binary labels for each stock for next 3 days
            for (let offset = 1; offset <= predictionHorizon; offset++) {
                const futureDate = this.dates[dateIdx + offset];
                this.symbols.forEach(symbol => {
                    const futureData = this.pivotedData[futureDate][symbol];
                    const currentClose = currentClosePrices[symbol];
                    
                    if (futureData && currentClose !== undefined) {
                        const futureClose = futureData.Close;
                        targetLabels.push(futureClose > currentClose ? 1 : 0);
                    } else {
                        // Default to 0 if data missing
                        targetLabels.push(0);
                    }
                });
            }

            samples.push(inputSequence);
            targets.push(targetLabels);
        });

        if (samples.length === 0) {
            throw new Error('No valid samples created');
        }

        console.log(`Created ${samples.length} samples with ${targets[0].length} targets each`);

        // Convert to tensors
        const X = tf.tensor3d(samples);
        const y = tf.tensor2d(targets);

        // Split chronologically (80% train, 20% test)
        const splitIdx = Math.floor(samples.length * 0.8);
        
        this.X_train = X.slice(0, splitIdx);
        this.X_test = X.slice(splitIdx);
        this.y_train = y.slice(0, splitIdx);
        this.y_test = y.slice(splitIdx);

        console.log(`Training samples: ${this.X_train.shape[0]}, Test samples: ${this.X_test.shape[0]}`);

        // Clean up intermediate tensors
        tf.dispose([X, y]);
    }

    getData() {
        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            symbols: this.symbols
        };
    }

    dispose() {
        const tensors = [this.X_train, this.y_train, this.X_test, this.y_test];
        tensors.forEach(tensor => {
            if (tensor && !tensor.isDisposed) {
                tensor.dispose();
            }
        });
    }
}
