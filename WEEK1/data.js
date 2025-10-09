=== FILE 3: data.js ===
// Titanic Data Processing and Cleaning
class TitanicDataProcessor {
    constructor() {
        this.combinedData = [];
        this.cleanData = [];
    }

    // Parse CSV data from the provided strings
    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        
        return lines.slice(1).map(line => {
            const values = [];
            let current = '';
            let inQuotes = false;
            
            for (let char of line) {
                if (char === '"') {
                    inQuotes = !inQuotes;
                } else if (char === ',' && !inQuotes) {
                    values.push(current);
                    current = '';
                } else {
                    current += char;
                }
            }
            values.push(current);
            
            const row = {};
            headers.forEach((header, index) => {
                row[header.trim()] = values[index] ? values[index].trim().replace(/^"|"$/g, '') : '';
            });
            return row;
        });
    }

    // Combine all datasets
    combineDatasets(trainCSV, testCSV, genderSubCSV) {
        try {
            // Parse training data
            const trainData = this.parseCSV(trainCSV);
            
            // Parse test data and survival labels
            const testData = this.parseCSV(testCSV);
            const genderSubData = this.parseCSV(genderSubCSV);
            
            // Create a map of test passenger survival
            const testSurvivalMap = {};
            genderSubData.forEach(row => {
                testSurvivalMap[row.PassengerId] = parseInt(row.Survived);
            });
            
            // Combine test data with survival labels
            const testDataWithSurvival = testData.map(passenger => ({
                ...passenger,
                Survived: testSurvivalMap[passenger.PassengerId] || null
            }));
            
            // Combine all data
            this.combinedData = [...trainData, ...testDataWithSurvival];
            
            console.log(`Combined ${this.combinedData.length} passenger records`);
            return this.combinedData;
            
        } catch (error) {
            console.error('Error combining datasets:', error);
            throw error;
        }
    }

    // Clean and preprocess data
    cleanAndPreprocess() {
        if (this.combinedData.length === 0) {
            throw new Error('No data available. Please combine datasets first.');
        }

        this.cleanData = this.combinedData.map(passenger => {
            const cleaned = { ...passenger };
            
            // Convert numeric fields
            cleaned.PassengerId = parseInt(cleaned.PassengerId);
            cleaned.Survived = cleaned.Survived !== null ? parseInt(cleaned.Survived) : null;
            cleaned.Pclass = parseInt(cleaned.Pclass);
            cleaned.Age = cleaned.Age ? parseFloat(cleaned.Age) : null;
            cleaned.SibSp = parseInt(cleaned.SibSp);
            cleaned.Parch = parseInt(cleaned.Parch);
            cleaned.Fare = cleaned.Fare ? parseFloat(cleaned.Fare) : null;
            
            // Handle missing Age values (median imputation)
            if (!cleaned.Age) {
                const medianAge = this.calculateMedianAge();
                cleaned.Age = medianAge;
            }
            
            // Create Age groups
            if (cleaned.Age <= 12) cleaned.AgeGroup = 'Child';
            else if (cleaned.Age <= 59) cleaned.AgeGroup = 'Adult';
            else cleaned.AgeGroup = 'Senior';
            
            // Create FamilySize
            cleaned.FamilySize = cleaned.SibSp + cleaned.Parch + 1;
            
            // Create FamilyCategory
            if (cleaned.FamilySize === 1) cleaned.FamilyCategory = 'Alone';
            else if (cleaned.FamilySize <= 4) cleaned.FamilyCategory = 'Small';
            else cleaned.FamilyCategory = 'Large';
            
            // Categorize Fare into quartiles
            if (cleaned.Fare <= 7.91) cleaned.FareCategory = 'Low';
            else if (cleaned.Fare <= 14.454) cleaned.FareCategory = 'Medium';
            else if (cleaned.Fare <= 31) cleaned.FareCategory = 'High';
            else cleaned.FareCategory = 'Very High';
            
            // Extract Title from Name
            const titleMatch = cleaned.Name.match(/\s([A-Za-z]+)\./);
            cleaned.Title = titleMatch ? titleMatch[1] : 'Unknown';
            
            // Simplify titles
            if (['Mr', 'Miss', 'Mrs', 'Master'].includes(cleaned.Title)) {
                // Keep as is
            } else if (['Dr', 'Rev', 'Col', 'Major', 'Capt'].includes(cleaned.Title)) {
                cleaned.Title = 'Professional';
            } else {
                cleaned.Title = 'Nobility';
            }
            
            // Handle missing Embarked
            if (!cleaned.Embarked || cleaned.Embarked === '') {
                cleaned.Embarked = 'S'; // Most common port
            }
            
            return cleaned;
        });
        
        console.log(`Cleaned ${this.cleanData.length} passenger records`);
        return this.cleanData;
    }

    // Calculate median age for imputation
    calculateMedianAge() {
        const ages = this.combinedData
            .map(p => p.Age)
            .filter(age => age && !isNaN(parseFloat(age)))
            .map(age => parseFloat(age))
            .sort((a, b) => a - b);
        
        const mid = Math.floor(ages.length / 2);
        return ages.length % 2 !== 0 ? ages[mid] : (ages[mid - 1] + ages[mid]) / 2;
    }

    // Get survival statistics by factor
    getSurvivalStatsByFactor(factor) {
        const groups = {};
        
        this.cleanData.forEach(passenger => {
            if (passenger.Survived === null) return;
            
            const groupValue = passenger[factor];
            if (!groups[groupValue]) {
                groups[groupValue] = {
                    total: 0,
                    survived: 0,
                    percentage: 0
                };
            }
            
            groups[groupValue].total++;
            if (passenger.Survived === 1) {
                groups[groupValue].survived++;
            }
        });
        
        // Calculate percentages
        Object.keys(groups).forEach(group => {
            groups[group].percentage = (groups[group].survived / groups[group].total) * 100;
        });
        
        return groups;
    }

    // Get overall survival statistics
    getOverallStats() {
        const totalPassengers = this.cleanData.length;
        const survivors = this.cleanData.filter(p => p.Survived === 1).length;
        const survivalRate = (survivors / totalPassengers) * 100;
        
        return {
            totalPassengers,
            survivors,
            survivalRate: Math.round(survivalRate * 10) / 10
        };
    }

    // Get data for visualizations
    getVisualizationData() {
        const genderStats = this.getSurvivalStatsByFactor('Sex');
        const classStats = this.getSurvivalStatsByFactor('Pclass');
        const ageStats = this.getSurvivalStatsByFactor('AgeGroup');
        const fareStats = this.getSurvivalStatsByFactor('FareCategory');
        const embarkedStats = this.getSurvivalStatsByFactor('Embarked');
        const familyStats = this.getSurvivalStatsByFactor('FamilyCategory');
        const titleStats = this.getSurvivalStatsByFactor('Title');
        
        return {
            gender: genderStats,
            class: classStats,
            age: ageStats,
            fare: fareStats,
            embarked: embarkedStats,
            family: familyStats,
            title: titleStats
        };
    }
}

// Create global data processor instance
const dataProcessor = new TitanicDataProcessor();

// Initialize data when the page loads
function initializeData() {
    try {
        // Use the provided CSV data (embedded in the HTML)
        const trainCSV = document.querySelector('[data-file="train.csv"]')?.textContent || '';
        const testCSV = document.querySelector('[data-file="test.csv"]')?.textContent || '';
        const genderSubCSV = document.querySelector('[data-file="gender_submission.csv"]')?.textContent || '';
        
        if (!trainCSV || !testCSV || !genderSubCSV) {
            throw new Error('CSV data not found in the page');
        }
        
        // Combine and clean data
        dataProcessor.combineDatasets(trainCSV, testCSV, genderSubCSV);
        dataProcessor.cleanAndPreprocess();
        
        return dataProcessor.cleanData;
        
    } catch (error) {
        console.error('Error initializing data:', error);
        // Fallback to sample data for demonstration
        return generateSampleData();
    }
}

// Fallback sample data generator (for demonstration)
function generateSampleData() {
    console.log('Using sample data for demonstration');
    // This would generate sample data if real CSV parsing fails
    return [];
}
