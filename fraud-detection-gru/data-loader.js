// DataLoader: Handles CSV loading, parsing, and preprocessing for fraud detection
class DataLoader {
    constructor() {
        this.data = null;
        this.sequences = [];
        this.labels = [];
        this.trainSequences = null;
        this.trainLabels = null;
        this.testSequences = null;
        this.testLabels = null;
        this.featureMeans = null;
        this.featureStds = null;
    }

    // Load and parse CSV file
    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const csvText = event.target.result;
                    const lines = csvText.trim().split('\n');
                    const headers = lines[0].split(',').map(h => h.trim());
                    
                    // Validate required columns
                    const requiredColumns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud'];
                    const missingColumns = requiredColumns.filter(col => !headers.includes(col));
                    
                    if (missingColumns.length > 0) {
                        throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
                    }
                    
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
                    }
                    
                    console.log(`Loaded ${this.data.length} transactions`);
                    resolve(this.data);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    // Normalize features using z-score normalization
    normalizeFeatures() {
        const features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'];
        this.featureMeans = {};
        this.featureStds = {};
        
        // Calculate mean and std for each feature
        features.forEach(feature => {
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
    }

    // Create sequences for GRU model (time series data)
    createSequences(sequenceLength = 10) {
        this.sequences = [];
        this.labels = [];
        
        const normalizedFeatures = [
            'step_normalized', 'amount_normalized', 'oldbalanceOrg_normalized',
            'newbalanceOrig_normalized', 'oldbalanceDest_normalized', 'newbalanceDest_normalized'
        ];
        
        // Create overlapping sequences
        for (let i = sequenceLength; i < this.data.length; i++) {
            const sequence = [];
            for (let j = i - sequenceLength; j < i; j++) {
                const features = normalizedFeatures.map(feature => this.data[j][feature]);
                sequence.push(features);
            }
            
            this.sequences.push(sequence);
            this.labels.push(this.data[i].isFraud);
        }
        
        console.log(`Created ${this.sequences.length} sequences of length ${sequenceLength}`);
    }

    // Split data into training and testing sets
    splitData(testSplit = 0.2) {
        const splitIndex = Math.floor(this.sequences.length * (1 - testSplit));
        
        // Convert to tensors
        this.trainSequences = tf.tensor3d(this.sequences.slice(0, splitIndex));
        this.trainLabels = tf.tensor1d(this.labels.slice(0, splitIndex));
        this.testSequences = tf.tensor3d(this.sequences.slice(splitIndex));
        this.testLabels = tf.tensor1d(this.labels.slice(splitIndex));
        
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
        if (this.trainSequences) this.trainSequences.dispose();
        if (this.trainLabels) this.trainLabels.dispose();
        if (this.testSequences) this.testSequences.dispose();
        if (this.testLabels) this.testLabels.dispose();
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
