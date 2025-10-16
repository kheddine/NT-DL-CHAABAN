// data-loader.js
// Train-only normalization with 70/15/15 chronological split.

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
    const required = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close'];
    for (const h of required)
      if (!headers.includes(h)) throw new Error(`Missing column: ${h}`);

    const data = {}, syms = new Set(), dts = new Set();
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',');
      if (vals.length < 6) continue;
      const row = {};
      headers.forEach((h, j) => (row[h] = vals[j].trim()));
      const s = row.Symbol, d = row.Date;
      if (!s || !d) continue;
      if (!data[s]) data[s] = {};
      data[s][d] = {
        Open: +row.Open, High: +row.High, Low: +row.Low, Close: +row.Close
      };
      syms.add(s); dts.add(d);
    }

    this.symbols = Array.from(syms).sort().slice(0, 10);
    this.dates   = Array.from(dts).sort();
    this.data    = data;
    if (!this.symbols.length) throw new Error('No symbols found.');
    console.log(`Loaded ${this.symbols.length} stocks × ${this.dates.length} days`);
  }

  createSequences(seqLen = 30, horizon = 3) {
    const Xraw = [], Y = [], validDates = [];
    const syms = this.symbols;

    for (let i = seqLen; i < this.dates.length - horizon; i++) {
      const seq = [];
      let valid = true;

      // build one 30-day window
      for (let j = i - seqLen; j < i; j++) {
        const d = this.dates[j];
        const step = [];
        for (const s of syms) {
          const r = this.data[s][d];
          if (!r) { valid = false; break; }
          step.push(r.Open, r.High, r.Low, r.Close);
        }
        if (!valid) break;
        seq.push(step);
      }

      if (!valid) continue;

      // build 3-day binary targets
      const y = [];
      for (let h = 1; h <= horizon; h++) {
        const fut = this.dates[i + h];
        for (const s of syms) {
          const now = this.data[s][this.dates[i]];
          const nxt = this.data[s][fut];
          if (!now || !nxt) { valid = false; break; }
          y.push(nxt.Close > now.Close ? 1 : 0);
        }
      }

      if (valid) { Xraw.push(seq); Y.push(y); validDates.push(this.dates[i]); }
    }

    if (!Xraw.length) throw new Error('No valid sequences.');

    // chronological 70/15/15 split
    const n = Xraw.length;
    const nTrain = Math.floor(n * 0.70);
    const nVal   = Math.floor(n * 0.85);
    const train = Xraw.slice(0, nTrain);
    const val   = Xraw.slice(nTrain, nVal);
    const test  = Xraw.slice(nVal);

    // compute per-stock min/max from train only
    const minMax = {};
    syms.forEach((s, si) => {
      let mn = Infinity, mx = -Infinity;
      for (const w of train) {
        for (const step of w) {
          const base = si * 4;
          for (let k = 0; k < 4; k++) {
            const v = step[base + k];
            if (v < mn) mn = v; if (v > mx) mx = v;
          }
        }
      }
      minMax[s] = { min: mn, max: mx };
    });

    // helper to scale window sets
    const scale = (windows) => windows.map(w =>
      w.map(step => {
        const out = [];
        syms.forEach((s, si) => {
          const { min, max } = minMax[s];
          const base = si * 4;
          for (let k = 0; k < 4; k++) {
            const v = step[base + k];
            out.push((v - min) / Math.max(1e-9, (max - min)));
          }
        });
        return out;
      })
    );

    const X_train = tf.tensor3d(scale(train));
    const y_train = tf.tensor2d(Y.slice(0, nTrain));
    const X_val   = tf.tensor3d(scale(val));
    const y_val   = tf.tensor2d(Y.slice(nTrain, nVal));
    const X_test  = tf.tensor3d(scale(test));
    const y_test  = tf.tensor2d(Y.slice(nVal));

    console.log(`Sequences → Train:${nTrain}  Val:${nVal - nTrain}  Test:${n - nVal}`);
    return { X_train, y_train, X_val, y_val, X_test, y_test, symbols: syms };
  }
}
