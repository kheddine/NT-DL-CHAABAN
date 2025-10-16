// data-loader.js
// Loads CSV, normalizes Open/High/Low/Close, and builds 30-day sequences for RNN.

export class DataLoader {
  constructor() {
    this.symbols = [];
    this.dates = [];
    this.data = {};
  }

  async loadCSV(file) {
    const text = await file.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const expected = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close'];

    // Check headers
    for (const h of expected) {
      if (!headers.includes(h)) {
        throw new Error(`Missing column: ${h}`);
      }
    }

    const data = {};
    const symbols = new Set();
    const dates = new Set();

    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',');
      if (vals.length < 6) continue;

      const row = {};
      headers.forEach((h, j) => (row[h] = vals[j].trim()));
      const sym = row.Symbol;
      const date = row.Date;
      if (!sym || !date) continue;

      if (!data[sym]) data[sym] = {};
      data[sym][date] = {
        Open: parseFloat(row.Open),
        High: parseFloat(row.High),
        Low: parseFloat(row.Low),
        Close: parseFloat(row.Close),
      };
      symbols.add(sym);
      dates.add(date);
    }

    this.symbols = Array.from(symbols).sort().slice(0, 10);
    this.dates = Array.from(dates).sort();
    this.data = data;

    if (this.symbols.length === 0) throw new Error("No valid symbols found in CSV.");

    console.log(`Loaded ${this.symbols.length} symbols, ${this.dates.length} days.`);
  }

  normalize() {
    for (const sym of this.symbols) {
      const records = Object.values(this.data[sym]);
      const all = records.flatMap(r => [r.Open, r.High, r.Low, r.Close]);
      const min = Math.min(...all);
      const max = Math.max(...all);
      for (const d in this.data[sym]) {
        const v = this.data[sym][d];
        for (const k of ["Open", "High", "Low", "Close"]) {
          v[k] = (v[k] - min) / (max - min);
        }
      }
    }
    console.log("Normalization complete.");
  }

  createSequences(seqLen = 30, horizon = 3) {
    this.normalize();
    const X = [], Y = [], validDates = [];

    for (let i = seqLen; i < this.dates.length - horizon; i++) {
      const seq = [];
      let valid = true;

      // Input window
      for (let j = i - seqLen; j < i; j++) {
        const step = [];
        const d = this.dates[j];
        for (const sym of this.symbols) {
          const rec = this.data[sym][d];
          if (!rec) { valid = false; break; }
          step.push(rec.Open, rec.High, rec.Low, rec.Close);
        }
        if (!valid) break;
        seq.push(step);
      }

      if (!valid) continue;

      // Output labels
      const target = [];
      for (let off = 1; off <= horizon; off++) {
        const futureDate = this.dates[i + off];
        for (const sym of this.symbols) {
          const now = this.data[sym][this.dates[i]];
          const fut = this.data[sym][futureDate];
          if (!now || !fut) { valid = false; break; }
          target.push(fut.Close > now.Close ? 1 : 0);
        }
      }

      if (valid) {
        X.push(seq);
        Y.push(target);
        validDates.push(this.dates[i]);
      }
    }

    if (X.length === 0) throw new Error("No valid sequences could be built.");

    const split = Math.floor(X.length * 0.8);
    const X_train = tf.tensor3d(X.slice(0, split));
    const y_train = tf.tensor2d(Y.slice(0, split));
    const X_test = tf.tensor3d(X.slice(split));
    const y_test = tf.tensor2d(Y.slice(split));

    return { X_train, y_train, X_test, y_test, symbols: this.symbols, dates: validDates.slice(split) };
  }
}
