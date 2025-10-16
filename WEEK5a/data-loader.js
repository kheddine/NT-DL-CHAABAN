/**
 * data-loader.js
 * Utilities for loading local CSV files, validating historical stock data,
 * normalising values, and producing TensorFlow.js tensors ready for training.
 */

const REQUIRED_HEADERS = ["Date", "Symbol", "Open", "High", "Low", "Close"];
const INPUT_WINDOW = 30;
const FORECAST_HORIZON = 3;
const FEATURES_PER_STOCK = 4; // Open, High, Low, Close
const MAX_STOCKS = 10;

/**
 * Reads the provided File object into a string.
 * @param {File} file
 * @returns {Promise<string>}
 */
async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read CSV file."));
    reader.onload = () => resolve(reader.result);
    reader.readAsText(file);
  });
}

/**
 * Validates the header row of the CSV.
 * @param {string[]} headers
 */
function validateHeaders(headers) {
  const missing = REQUIRED_HEADERS.filter((col) => !headers.includes(col));
  if (missing.length) {
    throw new Error(`CSV is missing required columns: ${missing.join(", ")}`);
  }
}

/**
 * Parses a CSV string into a nested stock->date map.
 * @param {string} csv
 * @param {(message: string) => void} statusCb
 */
function parseCSV(csv, statusCb) {
  const rows = csv.trim().split(/\r?\n/);
  if (rows.length <= 1) {
    throw new Error("CSV appears to be empty.");
  }

  const headers = rows[0].split(",").map((h) => h.trim());
  validateHeaders(headers);

  const dateIdx = headers.indexOf("Date");
  const symbolIdx = headers.indexOf("Symbol");
  const openIdx = headers.indexOf("Open");
  const highIdx = headers.indexOf("High");
  const lowIdx = headers.indexOf("Low");
  const closeIdx = headers.indexOf("Close");

  const data = new Map();

  for (let i = 1; i < rows.length; i++) {
    const parts = rows[i].split(",");
    if (parts.length < headers.length) continue;

    const date = parts[dateIdx]?.trim();
    const symbol = parts[symbolIdx]?.trim();
    const open = Number(parts[openIdx]);
    const high = Number(parts[highIdx]);
    const low = Number(parts[lowIdx]);
    const close = Number(parts[closeIdx]);

    if (!date || !symbol || [open, high, low, close].some((v) => Number.isNaN(v))) {
      continue; // Skip malformed rows
    }

    if (!data.has(symbol)) {
      data.set(symbol, new Map());
    }

    data.get(symbol).set(date, { Open: open, High: high, Low: low, Close: close });
  }

  const symbols = Array.from(data.keys()).slice(0, MAX_STOCKS);
  if (symbols.length < MAX_STOCKS) {
    statusCb?.(
      `Warning: Found only ${symbols.length} symbols. Proceeding with available data.`
    );
  }

  // Filter to ensure chronological ordering and intersection of dates
  let sharedDates = null;
  const symbolDateMap = new Map();
  for (const symbol of symbols) {
    const entries = Array.from(data.get(symbol).entries()).sort(
      (a, b) => new Date(a[0]) - new Date(b[0])
    );
    const dates = entries.map(([date]) => date);
    const dateSet = new Set(dates);
    const valueMap = new Map(entries);
    symbolDateMap.set(symbol, { dates, dateSet, valueMap });
    if (sharedDates === null) {
      sharedDates = new Set(dates);
    } else {
      sharedDates = new Set([...sharedDates].filter((d) => dateSet.has(d)));
    }
  }

  if (!sharedDates || sharedDates.size < INPUT_WINDOW + FORECAST_HORIZON) {
    throw new Error("Insufficient overlapping date coverage across selected symbols.");
  }

  const orderedDates = Array.from(sharedDates).sort((a, b) => new Date(a) - new Date(b));
  return { symbols, orderedDates, symbolDateMap };
}

/**
 * Builds training/validation/test splits and tensors.
 * @param {string} csv
 * @param {(message: string) => void} statusCb
 */
function buildTensors(csv, statusCb) {
  const { symbols, orderedDates, symbolDateMap } = parseCSV(csv, statusCb);
  const numStocks = symbols.length;
  const featureSize = numStocks * FEATURES_PER_STOCK;

  const totalSamples = orderedDates.length - INPUT_WINDOW - FORECAST_HORIZON + 1;
  if (totalSamples <= 0) {
    throw new Error("Not enough samples to build training sequences.");
  }

  const inputs = new Array(totalSamples);
  const labels = new Array(totalSamples);

  for (let sampleIdx = 0; sampleIdx < totalSamples; sampleIdx++) {
    const windowStart = sampleIdx;
    const windowEnd = sampleIdx + INPUT_WINDOW; // exclusive
    const baseDateIdx = windowEnd - 1;

    const window = new Array(INPUT_WINDOW);
    for (let step = 0; step < INPUT_WINDOW; step++) {
      const date = orderedDates[windowStart + step];
      const featureRow = new Array(featureSize);
      for (let s = 0; s < numStocks; s++) {
        const symbol = symbols[s];
        const entry = symbolDateMap.get(symbol).valueMap.get(date);
        if (!entry) {
          throw new Error(`Missing data for ${symbol} on ${date}.`);
        }
        const offset = s * FEATURES_PER_STOCK;
        featureRow[offset] = entry.Open;
        featureRow[offset + 1] = entry.High;
        featureRow[offset + 2] = entry.Low;
        featureRow[offset + 3] = entry.Close;
      }
      window[step] = featureRow;
    }
    inputs[sampleIdx] = window;

    const baseDate = orderedDates[baseDateIdx];
    const labelsRow = new Array(numStocks * FORECAST_HORIZON);
    for (let s = 0; s < numStocks; s++) {
      const symbol = symbols[s];
      const symbolInfo = symbolDateMap.get(symbol);
      const baseClose = symbolInfo.valueMap.get(baseDate)?.Close;
      if (baseClose === undefined) {
        throw new Error(`Missing closing price for ${symbol} on ${baseDate}.`);
      }
      for (let horizon = 1; horizon <= FORECAST_HORIZON; horizon++) {
        const futureDate = orderedDates[baseDateIdx + horizon];
        const futureEntry = symbolInfo.valueMap.get(futureDate);
        if (!futureEntry) {
          throw new Error(`Missing future data for ${symbol} on ${futureDate}.`);
        }
        const labelIdx = s * FORECAST_HORIZON + (horizon - 1);
        labelsRow[labelIdx] = futureEntry.Close > baseClose ? 1 : 0;
      }
    }
    labels[sampleIdx] = labelsRow;
  }

  const trainCount = Math.floor(totalSamples * 0.7);
  const remaining = totalSamples - trainCount;
  let valCount = Math.floor(totalSamples * 0.15);
  if (remaining - valCount <= 0) {
    // Fallback to 80/20 split if validation set would be empty
    valCount = 0;
  }
  const testCount = totalSamples - trainCount - valCount;

  if (trainCount === 0 || testCount === 0) {
    throw new Error("Unable to split data into train/test sets. Provide more samples.");
  }

  statusCb?.(
    `Prepared ${totalSamples} samples. Train: ${trainCount}, Validation: ${valCount}, Test: ${testCount}.`
  );

  const trainInputs = inputs.slice(0, trainCount);
  const trainLabels = labels.slice(0, trainCount);
  const valInputs = valCount > 0 ? inputs.slice(trainCount, trainCount + valCount) : [];
  const valLabels = valCount > 0 ? labels.slice(trainCount, trainCount + valCount) : [];
  const testInputs = inputs.slice(trainCount + valCount);
  const testLabels = labels.slice(trainCount + valCount);

  const mins = Array.from({ length: numStocks }, () =>
    Array(FEATURES_PER_STOCK).fill(Number.POSITIVE_INFINITY)
  );
  const maxs = Array.from({ length: numStocks }, () =>
    Array(FEATURES_PER_STOCK).fill(Number.NEGATIVE_INFINITY)
  );

  for (const window of trainInputs) {
    for (const row of window) {
      for (let s = 0; s < numStocks; s++) {
        for (let f = 0; f < FEATURES_PER_STOCK; f++) {
          const idx = s * FEATURES_PER_STOCK + f;
          const value = row[idx];
          if (value < mins[s][f]) mins[s][f] = value;
          if (value > maxs[s][f]) maxs[s][f] = value;
        }
      }
    }
  }

  const normaliseWindow = (window) => {
    return window.map((row) => {
      const normalised = new Array(featureSize);
      for (let s = 0; s < numStocks; s++) {
        for (let f = 0; f < FEATURES_PER_STOCK; f++) {
          const idx = s * FEATURES_PER_STOCK + f;
          const min = mins[s][f];
          const max = maxs[s][f];
          const range = max - min;
          normalised[idx] = range === 0 ? 0 : (row[idx] - min) / range;
        }
      }
      return normalised;
    });
  };

  const normaliseSet = (set) => set.map((window) => normaliseWindow(window));

  const tensors = {
    X_train: tf.tensor3d(normaliseSet(trainInputs), [trainCount, INPUT_WINDOW, featureSize]),
    y_train: tf.tensor2d(trainLabels, [trainCount, numStocks * FORECAST_HORIZON]),
    X_val:
      valCount > 0
        ? tf.tensor3d(normaliseSet(valInputs), [valCount, INPUT_WINDOW, featureSize])
        : null,
    y_val:
      valCount > 0 ? tf.tensor2d(valLabels, [valCount, numStocks * FORECAST_HORIZON]) : null,
    X_test: tf.tensor3d(normaliseSet(testInputs), [testCount, INPUT_WINDOW, featureSize]),
    y_test: tf.tensor2d(testLabels, [testCount, numStocks * FORECAST_HORIZON]),
    symbols,
    meta: {
      numStocks,
      featureSize,
      counts: { train: trainCount, val: valCount, test: testCount },
    },
  };

  return tensors;
}

/**
 * Public API: load CSV file and prepare tensors.
 * @param {File} file
 * @param {{ onStatus?: (msg: string) => void }} options
 */
export async function loadCSVFile(file, options = {}) {
  const { onStatus } = options;
  if (!file) {
    throw new Error("No CSV file provided.");
  }
  onStatus?.("Reading CSV file...");
  const text = await readFileAsText(file);
  onStatus?.("Parsing records and building sequences...");
  return buildTensors(text, onStatus);
}

export const DataConstants = {
  INPUT_WINDOW,
  FORECAST_HORIZON,
  FEATURES_PER_STOCK,
};
