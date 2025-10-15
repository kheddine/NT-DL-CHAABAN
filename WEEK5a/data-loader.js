// data-loader.js
// Loads CSV, normalizes Open/High/Low/Close, builds 30-day sequences and 3-day-ahead binary targets.

export class DataLoader {
  constructor() {
    this.symbols = [];
    this.dates = [];
    this.data = {};
  }

  async loadCSV(file) {
    const text = await file.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');

    const data = {};
    const symbols = new Set();
    const dates = new Set();

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length !== headers.length) continue;
      const row = {};
      headers.forEach((h, idx) => row[h.trim()] = values[idx].trim());
      const sym = row.Symbol;
      const date = row.Date;
      if (!sym || !date) continue;
      if (!data[sym]) data[sym] = {};
      data[sym][date] = {
        Open: parseFloat(row.Open),
        High: parseFloat(row.High),
        Low: parseFloat(row.Low),
        Close: parseFloat(row.Close)
      };
      symbols.add(sym);
      dates.add(date);
    }

    this.symbols = Array.from(symbols).sort().slice(0, 10);
    this.dates = Array.from(dates).sort();
    this.data = data;
    console.log(`Loaded ${this.symbols.length} stocks, ${this.dates.length} days`);
  }

  normalize() {
    const minMax = {};
    for (const sym of this.symbols) {
      const rows = Object.values(this.data[sym]);
      const allVals = rows.flatMap(r => [r.Open, r.High, r.Low, r.Close]);
      const min = Math.min(...allVals);
      const max = Math.max(...allVals);
      minMax[sym] = { min, max };
      for (const d in this.data[sym]) {
        const v = this.data[sym][d];
        for (const k of ["Open", "High", "Low", "Close"]) {
          v[k] = (v[k] - min) / (max - min);
        }
      }
    }
    console.log("Data normalized");
  }

  createSequences(seqLen = 30, horizon = 3) {
    this.normalize();
    const X = [], Y = [], validDates = [];
    for (let i = seqLen; i < this.dates.length - horizon; i++) {
      const window = [];
      let valid = true;
      for (let j = i - seqLen; j < i; j++) {
        const d = this.dates[j];
        const step = [];
        for (const sym of this.symbols) {
          const r = this.data[sym][d];
          if (!r) { valid = false; break; }
          step.push(r.Open, r.High, r.Low, r.Close);
        }
        if (!valid) break;
        window.push(step);
      }
      if (!valid) continue;

      const target = [];
      for (let offset = 1; offset <= horizon; offset++) {
        const future = this.dates[i + offset];
        for (const sym of this.symbols) {
          const now = this.data[sym][this.dates[i]];
          const fut = this.data[sym][future];
          if (!now || !fut) { valid = false; break; }
          target.push(fut.Close > now.Close ? 1 : 0);
        }
      }
      if (valid) {
        X.push(window);
        Y.push(target);
        validDates.push(this.dates[i]);
      }
    }

    const split = Math.floor(X.length * 0.8);
    const X_train = tf.tensor3d(X.slice(0, split));
    const y_train = tf.tensor2d(Y.slice(0, split));
    const X_test = tf.tensor3d(X.slice(split));
    const y_test = tf.tensor2d(Y.slice(split));

    return { X_train, y_train, X_test, y_test, symbols: this.symbols, dates: validDates.slice(split) };
  }
}
