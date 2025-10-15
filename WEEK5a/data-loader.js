// data-loader.js
// Handles CSV parsing, normalization, and dataset preparation for RNN training.
// Uses Open, High, Low, Close over the last 30 days for 10 stocks.

export class DataLoader {
  constructor() {
    this.symbols = [];
    this.minMax = {}; // per stock normalization
  }

  async loadCSV(file) {
    const text = await file.text();
    const lines = text.trim().split('\n');
    const header = lines[0].split(',');
    const rows = lines.slice(1).map(line => line.split(','));
    return { header, rows };
  }

  async prepareData(file) {
    const { header, rows } = await this.loadCSV(file);
    const data = {};

    // Group by symbol
    for (const r of rows) {
      const [date, symbol, open, high, low, close] = r;
      if (!data[symbol]) data[symbol] = [];
      data[symbol].push({
        date: new Date(date),
        open: parseFloat(open),
        high: parseFloat(high),
        low: parseFloat(low),
        close: parseFloat(close),
      });
    }

    // Sort by date and normalize per symbol
    this.symbols = Object.keys(data).sort();
    for (const sym of this.symbols) {
      data[sym].sort((a, b) => a.date - b.date);
      const all = data[sym];
      const values = all.flatMap(d => [d.open, d.high, d.low, d.close]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      this.minMax[sym] = { min, max };
      for (const d of all) {
        d.open = (d.open - min) / (max - min);
        d.high = (d.high - min) / (max - min);
        d.low = (d.low - min) / (max - min);
        d.close = (d.close - min) / (max - min);
      }
    }

    // Create sliding windows
    const seqLen = 30;
    const horizon = 3;
    const X = [];
    const Y = [];
    let dates = data[this.symbols[0]].map(d => d.date);

    for (let i = 0; i < dates.length - seqLen - horizon; i++) {
      const xWindow = [];
      const yLabels = [];
      for (const sym of this.symbols) {
        const series = data[sym];
        const window = series.slice(i, i + seqLen);
        const baseClose = series[i + seqLen - 1].close;
        const features = window.flatMap(d => [d.open, d.high, d.low, d.close]);
        xWindow.push(...features);

        for (let h = 1; h <= horizon; h++) {
          const future = series[i + seqLen - 1 + h].close;
          yLabels.push(future > baseClose ? 1 : 0);
        }
      }
      X.push(xWindow);
      Y.push(yLabels);
    }

    const split = Math.floor(0.8 * X.length);
    const X_train = tf.tensor3d(X.slice(0, split), [split, seqLen, 4 * this.symbols.length]);
    const y_train = tf.tensor2d(Y.slice(0, split), [split, 3 * this.symbols.length]);
    const X_test = tf.tensor3d(X.slice(split), [X.length - split, seqLen, 4 * this.symbols.length]);
    const y_test = tf.tensor2d(Y.slice(split), [Y.length - split, 3 * this.symbols.length]);

    return { X_train, y_train, X_test, y_test, symbols: this.symbols };
  }
}
