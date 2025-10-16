// data-loader.js
class DataLoader {
    constructor() {
        this.data = {};
        this.symbols = [];
        this.sequences = [];
        this.labels = [];
        this.normalizationParams = {};
    }

    // Load and parse CSV file
    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const csvText = e.target.result;
                    this.parseCSV(csvText);
                    resolve(this.data);
                } catch (error) {
                    reject(new Error(`CSV parsing failed: ${error.message}`));
                }
            };
            
            reader.onerror = () => reject(new Error('File reading failed'));
            reader.readAsText(file);
        });
    }

    // Parse CSV text into structured data
    parseCSV(csvText) {
        this.data = {};
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Validate required columns
        const requiredColumns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close'];
        const missingColumns = requiredColumns.filter(col => !headers.includes(col));
        if (missingColumns.length > 0) {
            throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
        }

        // Parse data rows
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });

            // Skip rows with invalid numeric values
            const numericFields = ['Open', 'High', 'Low', 'Close'];
            const hasNaN = numericFields.some(field => isNaN(parseFloat(row[field])));
            if (hasNaN) continue;

            const symbol = row.Symbol;
            const date = row.Date;

            // Initialize symbol data structure if needed
            if (!this.data[symbol]) {
                this.data[symbol] = {};
            }

            // Store price data
            this.data[symbol][date] = {
                Open: parseFloat(row.Open),
                High: parseFloat(row.High),
                Low: parseFloat(row.Low),
                Close: parseFloat(row.Close)
            };
        }

        // Get first 10 unique symbols
        this.symbols = Object.keys(this.data).slice(0, 10);
        if (this.symbols.length === 0) {
            throw new Error('No valid stock data found in CSV');
        }

        console.log(`Loaded data for ${this.symbols.length} symbols:`, this.symbols);
    }

    // Build sequences for training
    buildSequences(windowSize = 30, predictionOffset = 3) {
        this.sequences = [];
        this.labels = [];
        
        // Get all unique dates across all symbols
        const allDates = new Set();
        this.symbols.forEach(symbol => {
            Object.keys(this.data[symbol]).forEach(date => allDates.add(date));
        });
        
        const sortedDates = Array.from(allDates).sort();
        
        // For each possible starting point
        for (let i = 0; i < sortedDates.length - windowSize - predictionOffset; i++) {
            const sequenceStart = i;
            const sequenceEnd = i + windowSize;
            const predictionDate = i + windowSize + predictionOffset - 1;
            
            // Check if we have all required data points
            let validSequence = true;
            const sequenceFeatures = [];
            
            // Build features for all symbols in this window
            for (let j = sequenceStart; j < sequenceEnd; j++) {
                const date = sortedDates[j];
                const features = [];
                
                for (const symbol of this.symbols) {
                    if (!this.data[symbol][date]) {
                        validSequence = false;
                        break;
                    }
                    
                    const prices = this.data[symbol][date];
                    features.push(
                        prices.Open,
                        prices.High, 
                        prices.Low,
                        prices.Close
                    );
                }
                
                if (!validSequence) break;
                sequenceFeatures.push(features);
            }
            
            if (!validSequence) continue;
            
            // Build labels for all symbols
            const sequenceLabels = [];
            const currentDate = sortedDates[sequenceEnd - 1];
            const futureDate = sortedDates[predictionDate];
            
            for (const symbol of this.symbols) {
                if (!this.data[symbol][currentDate] || !this.data[symbol][futureDate]) {
                    validSequence = false;
                    break;
                }
                
                const currentClose = this.data[symbol][currentDate].Close;
                const futureClose = this.data[symbol][futureDate].Close;
                const label = futureClose > currentClose ? 1 : 0;
                sequenceLabels.push(label);
            }
            
            if (!validSequence) continue;
            
            this.sequences.push(sequenceFeatures);
            this.labels.push(sequenceLabels);
        }
        
        console.log(`Built ${this.sequences.length} sequences`);
    }

    // Normalize data using min-max scaling
    normalizeData() {
        this.normalizationParams = {};
        
        // Calculate min/max for each symbol's features using only training data
        const trainSize = Math.floor(this.sequences.length * 0.7);
        const trainSequences = this.sequences.slice(0, trainSize);
        
        // Initialize min/max trackers
        this.symbols.forEach((symbol, symbolIndex) => {
            this.normalizationParams[symbol] = {
                min: [Infinity, Infinity, Infinity, Infinity], // Open, High, Low, Close
                max: [-Infinity, -Infinity, -Infinity, -Infinity]
            };
        });
        
        // Find min/max values
        trainSequences.forEach(sequence => {
            sequence.forEach(timestep => {
                this.symbols.forEach((symbol, symbolIndex) => {
                    const featureStart = symbolIndex * 4;
                    const features = timestep.slice(featureStart, featureStart + 4);
                    
                    features.forEach((value, featureIndex) => {
                        const params = this.normalizationParams[symbol];
                        params.min[featureIndex] = Math.min(params.min[featureIndex], value);
                        params.max[featureIndex] = Math.max(params.max[featureIndex], value);
                    });
                });
            });
        });
        
        // Apply normalization to all sequences
        this.sequences = this.sequences.map(sequence => 
            sequence.map(timestep => {
                const normalizedTimestep = [];
                
                this.symbols.forEach((symbol, symbolIndex) => {
                    const featureStart = symbolIndex * 4;
                    const features = timestep.slice(featureStart, featureStart + 4);
                    const params = this.normalizationParams[symbol];
                    
                    features.forEach((value, featureIndex) => {
                        const min = params.min[featureIndex];
                        const max = params.max[featureIndex];
                        const normalized = max === min ? 0 : (value - min) / (max - min);
                        normalizedTimestep.push(normalized);
                    });
                });
                
                return normalizedTimestep;
            })
        );
    }

    // Split data into train/validation/test sets
    splitData() {
        const totalSamples = this.sequences.length;
        const trainSize = Math.floor(totalSamples * 0.7);
        const valSize = Math.floor(totalSamples * 0.15);
        
        const X_train = this.sequences.slice(0, trainSize);
        const y_train = this.labels.slice(0, trainSize);
        
        const X_val = this.sequences.slice(trainSize, trainSize + valSize);
        const y_val = this.labels.slice(trainSize, trainSize + valSize);
        
        const X_test = this.sequences.slice(trainSize + valSize);
        const y_test = this.labels.slice(trainSize + valSize);
        
        console.log(`Data split - Train: ${X_train.length}, Val: ${X_val.length}, Test: ${X_test.length}`);
        
        return {
            X_train: tf.tensor3d(X_train),
            y_train: tf.tensor2d(y_train),
            X_val: X_val.length > 0 ? tf.tensor3d(X_val) : null,
            y_val: y_val.length > 0 ? tf.tensor2d(y_val) : null,
            X_test: tf.tensor3d(X_test),
            y_test: tf.tensor2d(y_test),
            symbols: this.symbols
        };
    }

    // Main processing pipeline
    async processData(file) {
        await this.loadCSV(file);
        this.buildSequences();
        this.normalizeData();
        return this.splitData();
    }
}

// Export for use in other modules
window.DataLoader = DataLoader;
