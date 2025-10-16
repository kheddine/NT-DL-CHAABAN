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
                row[header] = values[index];
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
        
        if (this.symbols.length !== 10) {
            console.warn(`Expected 10 stocks, found ${this.symbols.length}`);
        }

        // Pivot data: date × symbol → features
        this.pivotedData = {};
        this.dates.forEach(date => {
            this.pivotedData[date] = {};
            this.symbols.forEach(symbol => {
                const row = this.data.find(r => r.Date === date && r.Symbol === symbol);
                if (row) {
                    this.pivotedData[date][symbol] = {
                        Open: parseFloat(row.Open),
                        Close: parseFloat(row.Close)
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
            const opens = this.dates.map(date => this.pivotedData[date]?.[symbol]?.Open).filter(v => v != null);
            const closes = this.dates.map(date => this.pivotedData[date]?.[symbol]?.Close).filter(v => v != null);
            
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
                    this.normalizedData[date][symbol] = {
                        Open: (data.Open - this.minMax[symbol].Open.min) / 
                              (this.minMax[symbol].Open.max - this.minMax[symbol].Open.min),
                        Close: (data.Close - this.minMax[symbol].Close.min) / 
                               (this.minMax[symbol].Close.max - this.minMax[symbol].Close.min)
                    };
                }
            });
        });
    }

    createSamples(sequenceLength = 12, predictionHorizon = 3) {
        const samples = [];
        const targets = [];
        const validDates = this.dates.slice(sequenceLength, this.dates.length - predictionHorizon);

        validDates.forEach((date, dateIndex) => {
            const dateIdx = this.dates.indexOf(date);
            const inputSequence = [];
            
            // Get 12-day sequence
            for (let i = dateIdx - sequenceLength; i < dateIdx; i++) {
                const seqDate = this.dates[i];
                const features = [];
                
                this.symbols.forEach(symbol => {
                    const stockData = this.normalizedData[seqDate][symbol];
                    features.push(stockData.Open, stockData.Close);
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
                    const futureClose = this.pivotedData[futureDate][symbol].Close;
                    const currentClose = currentClosePrices[symbol];
                    targetLabels.push(futureClose > currentClose ? 1 : 0);
                });
            }

            samples.push(inputSequence);
            targets.push(targetLabels);
        });

        // Convert to tensors
        const X = tf.tensor3d(samples);
        const y = tf.tensor2d(targets);

        // Split chronologically (80% train, 20% test)
        const splitIdx = Math.floor(samples.length * 0.8);
        
        this.X_train = X.slice(0, splitIdx);
        this.X_test = X.slice(splitIdx);
        this.y_train = y.slice(0, splitIdx);
        this.y_test = y.slice(splitIdx);

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
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}
