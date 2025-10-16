// data-loader.js
// Loads CSV, builds sequences, train-only normalization (70/15/15 split)

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
    const req = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close'];
    for (const h of req) if (!headers.includes(h)) throw new Error(`Missing column: ${h}`);

    const data = {}, syms = new Set(), dts = new Set();
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',');
      if (vals.length < 6) continue;
      const row = {}; headers.forEach((h, j) => (row[h] = vals[j].trim()));
      const s = row.Symbol, d = row.Date;
      if (!s || !d) continue;
      if (!data[s]) data[s] = {};
      data[s][d] = { Open: +row.Open, High: +row.High, Low: +row.Low, Close: +row.Close };
      syms.add(s); dts.add(d);
    }

    this.symbols = Array.from(syms).sort().slice(0, 10);
    this.dates = Array.from(dts).sort();
    this.data = data;
    if (!this.symbols.length) throw new Error('No valid symbols in CSV');
    console.log(`Loaded ${this.symbols.length} symbols × ${this.dates.length} days`);
  }

  createSequences(seqLen = 30, horizon = 3) {
    const Xraw = [], Y = [], syms = this.symbols;

    for (let i = seqLen; i < this.dates.length - horizon; i++) {
      const seq = []; let valid = true;
      for (let j = i - seqLen; j < i; j++) {
        const d = this.dates[j]; const step = [];
        for (const s of syms) {
          const r = this.data[s][d];
          if (!r || Object.values(r).some(v => isNaN(v))) { valid = false; break; }
          step.push(r.Open, r.High, r.Low, r.Close);
        }
        if (!valid) break;
        seq.push(step);
      }
      if (!valid) continue;

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
      if (valid) { Xraw.push(seq); Y.push(y); }
    }

    if (!Xraw.length) throw new Error('No valid sequences built');

    const n = Xraw.length, nTrain = Math.floor(n * 0.7), nVal = Math.floor(n * 0.85);
    const Xtr = Xraw.slice(0, nTrain), Xv = Xraw.slice(nTrain, nVal), Xte = Xraw.slice(nVal);

    // compute train-only min/max per stock
    const minMax = {};
    syms.forEach((s, si) => {
      let mn = Infinity, mx = -Infinity;
      for (const w of Xtr)
        for (const st of w) {
          const base = si * 4;
          for (let k = 0; k < 4; k++) {
            const v = st[base + k];
            if (!isNaN(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
          }
        }
      if (!isFinite(mn) || !isFinite(mx) || mn === mx) { mn = 0; mx = 1; }
      minMax[s] = { min: mn, max: mx };
    });

    const scale = arr => arr.map(w => w.map(st => {
      const out = [];
      syms.forEach((s, si) => {
        const { min, max } = minMax[s]; const base = si * 4;
        for (let k = 0; k < 4; k++) {
          const v = st[base + k];
          out.push((v - min) / Math.max(1e-9, max - min));
        }
      });
      return out;
    }));

    const X_train = tf.tensor3d(scale(Xtr));
    const y_train = tf.tensor2d(Y.slice(0, nTrain));
    const X_val = tf.tensor3d(scale(Xv));
    const y_val = tf.tensor2d(Y.slice(nTrain, nVal));
    const X_test = tf.tensor3d(scale(Xte));
    const y_test = tf.tensor2d(Y.slice(nVal));

    console.log(`Sequences → Train:${nTrain}  Val:${nVal - nTrain}  Test:${n - nVal}`);
    return { X_train, y_train, X_val, y_val, X_test, y_test, symbols: syms };
  }
}
