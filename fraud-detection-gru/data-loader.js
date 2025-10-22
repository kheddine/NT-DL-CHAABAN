// DataLoader: Handles CSV loading, parsing, and preprocessing for fraud detection
class DataLoader {
    constructor(processCallback) {
        this.data = null;
        this.sequences = [];
        this.labels = [];
        this.trainSequences = null;
        this.trainLabels = null;
        this.testSequences = null;
        this.testLabels = null;
        this.featureMeans = null;
        this.featureStds = null;
        this.processCallback = processCallback;
    }

    // Update process window
    updateProcess(message, type = 'info', isActive = false) {
        if (this.processCallback) {
            this.processCallback(message, type, isActive);
        }
    }

    // Load and parse CSV file
    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            this.updateProcess('Starting CSV file loading...', 'info', true);
            
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    this.updateProcess('CSV file read successfully, starting parsing...', 'info', true);
                    const csvText = event.target.result;
                    const lines = csvText.trim().split('\n');
                    const headers = lines[0].split(',').map(h => h.trim());
                    
                    this.updateProcess(`Found ${headers.length} columns: ${headers.join(', ')}`, 'info');
                    
                    // Validate required columns
                    const requiredColumns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud'];
                    const missingColumns = requiredColumns.filter(col => !headers.includes(col));
                    
                    if (missingColumns.length > 0) {
                        this.updateProcess(`Missing required columns: ${missingColumns.join(', ')}`, 'error');
                        throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
                    }
                    
                    this.updateProcess('All required columns found, parsing data rows...', 'info', true);
                    
                    // Parse data rows
                    this.data = [];
                    for (let i = 1; i < lines.length; i++) {
                        const values = lines[i].split(',');
                        if (values.length === headers.length) {
                            const row = {};
                            headers.forEach((header, index) => {
                                row[header] = parseFloat(values[index]) || 0;
                            });
                            this.data.push(row);
                        }
                        
                        // Update progress for large files
                        if (i % 1000 === 0) {
                            this.updateProcess(`Parsed ${i} rows...`, 'info');
                        }
                    }
                    
                    this.updateProcess(`Successfully loaded ${this.data.length} transactions`, 'success');
                    console.log(`Loaded ${this.data.length} transactions`);
                    resolve(this.data);
                } catch (error) {
                    this.updateProcess(`Error during CSV parsing: ${error.message}`, 'error');
                    reject(error);
                }
            };
            
            reader.onerror = () => {
                this.updateProcess('Failed to read file', 'error');
                reject(new Error('Failed to read file'));
            };
            
            reader.onprogress = (event) => {
                if (event.lengthComputable) {
                    const percent = (event.loaded / event.total) * 100;
                    this.updateProcess(`Reading file: ${percent.toFixed(1)}%`, 'info');
                }
            };
            
            reader.readAsText(file);
        });
    }

    // Normalize features using z-score normalization
    normalizeFeatures() {
        this.updateProcess('Starting feature normalization...', 'info', true);
        const features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'];
        this.featureMeans = {};
        this.featureStds = {};
        
        // Calculate mean and std for each feature
        features.forEach((feature, index) => {
            this.updateProcess(`Processing feature ${index + 1}/${features.length}: ${feature}`, 'info', true);
            const values = this.data.map(row => row[feature]);
            this.featureMeans[feature] = values.reduce((a, b) => a + b, 0) / values.length;
            const squaredDiffs = values.map(x => Math.pow(x - this.featureMeans[feature], 2));
            this.featureStds[feature] = Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
            
            // Normalize data
            this.data.forEach(row => {
                row[`${feature}_normalized`] = 
                    (this.featureStds[feature] !== 0) 
                    ? (row[feature] - this.featureMeans[feature]) / this.featureStds[feature]
                    : 0;
            });
        });
        
        this.updateProcess('Feature normalization completed successfully', 'success');
    }

    // Create sequences for GRU model (time series data)
    createSequences(sequenceLength = 10) {
        this.updateProcess(`Creating sequences with length ${sequenceLength}...`, 'info', true);
        this.sequences = [];
        this.labels = [];
        
        const normalizedFeatures = [
            'step_normalized', 'amount_normalized', 'oldbalanceOrg_normalized',
            'newbalanceOrig_normalized', 'oldbalanceDest_normalized', 'newbalanceDest_normalized'
        ];
        
        this.updateProcess(`Using ${normalizedFeatures.length} normalized features`, 'info');
        
        // Create overlapping sequences
        for (let i = sequenceLength; i < this.data.length; i++) {
            const sequence = [];
            for (let j = i - sequenceLength; j < i; j++) {
                const features = normalizedFeatures.map(feature => this.data[j][feature]);
                sequence.push(features);
            }
            
            this.sequences.push(sequence);
            this.labels.push(this.data[i].isFraud);
            
            // Update progress for large datasets
            if (i % 1000 === 0) {
                this.updateProcess(`Created ${this.sequences.length} sequences...`, 'info');
            }
        }
        
        this.updateProcess(`Created ${this.sequences.length} sequences of length ${sequenceLength}`, 'success');
        console.log(`Created ${this.sequences.length} sequences of length ${sequenceLength}`);
    }

    // Split data into training and testing sets
    splitData(testSplit = 0.2) {
        this.updateProcess(`Splitting data (${(1-testSplit)*100}% train, ${testSplit*100}% test)...`, 'info', true);
        const splitIndex = Math.floor(this.sequences.length * (1 - testSplit));
        
        this.updateProcess('Converting sequences to tensors...', 'info', true);
        
        // Convert to tensors
        this.trainSequences = tf.tensor3d(this.sequences.slice(0, splitIndex));
        this.trainLabels = tf.tensor1d(this.labels.slice(0, splitIndex));
        this.testSequences = tf.tensor3d(this.sequences.slice(splitIndex));
        this.testLabels = tf.tensor1d(this.labels.slice(splitIndex));
        
        this.updateProcess(`Training sequences: ${this.trainSequences.shape[0]}`, 'info');
        this.updateProcess(`Testing sequences: ${this.testSequences.shape[0]}`, 'info');
        this.updateProcess('Data splitting and tensor conversion completed', 'success');
        
        console.log(`Training sequences: ${this.trainSequences.shape[0]}`);
        console.log(`Testing sequences: ${this.testSequences.shape[0]}`);
        
        return {
            trainSequences: this.trainSequences,
            trainLabels: this.trainLabels,
            testSequences: this.testSequences,
            testLabels: this.testLabels
        };
    }

    // Clean up tensors to prevent memory leaks
    dispose() {
        this.updateProcess('Cleaning up tensors and memory...', 'info');
        if (this.trainSequences) this.trainSequences.dispose();
        if (this.trainLabels) this.trainLabels.dispose();
        if (this.testSequences) this.testSequences.dispose();
        if (this.testLabels) this.testLabels.dispose();
        this.updateProcess('Memory cleanup completed', 'success');
    }

    // Get data statistics
    getDataInfo() {
        if (!this.data) return null;
        
        const fraudCount = this.data.filter(row => row.isFraud === 1).length;
        const nonFraudCount = this.data.length - fraudCount;
        
        return {
            totalSamples: this.data.length,
            fraudCount,
            nonFraudCount,
            fraudPercentage: (fraudCount / this.data.length * 100).toFixed(2)
        };
    }
}

export default DataLoader;
