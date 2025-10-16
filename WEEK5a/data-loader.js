// data-loader.js
// CSV ingest → raw sequence build → train-only normalization → tensors
// Safe against missing/NaN rows and zero-length splits.

export class DataLoader {
  constructor() {
    this.symbols = [];
    this.dates = [];
    this.data = {};
  }

  async loadCSV(file) {
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(l => l.trim().length);
    if (lines.length < 2) throw new Error("CSV appears empty.");

    const headers = lines[0].split(",").map(h => h.trim());
    const required = ["Date","Symbol","Open","High","Low","Close"];
    for (const col of required) {
      if (!headers.includes(col)) {
        throw new Error(`Missing column: ${col}. Found: ${headers.join(", ")}`);
      }
    }

    const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
    const data = {};
    const syms = new Set();
    const dts  = new Set();

    for (let i=1;i<lines.length;i++){
      const parts = lines[i].split(","); if (parts.length < headers.length) continue;
      const date  = parts[idx.Date]?.trim();
      const sym   = parts[idx.Symbol]?.trim();
      const open  = parseFloat(parts[idx.Open]);
      const high  = parseFloat(parts[idx.High]);
      const low   = parseFloat(parts[idx.Low]);
      const close = parseFloat(parts[idx.Close]);

      if (!date || !sym) continue;
      if ([open,high,low,close].some(v => Number.isNaN(v))) continue;

      if (!data[sym]) data[sym] = {};
      data[sym][date] = { Open: open, High: high, Low: low, Close: close };
      syms.add(sym); dts.add(date);
    }

    this.symbols = Array.from(syms).sort().slice(0, 10);
    this.dates   = Array.from(dts).sort();
    this.data    = data;

    if (this.symbols.length === 0) throw new Error("No valid symbols after parsing.");
    if (this.dates.length < 40)    throw new Error("Not enough dates for 30-day windows.");

    console.log(`Loaded: ${this.symbols.length} symbols × ${this.dates.length} days`);
  }

  createSequences(seqLen = 30, horizon = 3) {
    const Xraw = [];
    const Y    = [];
    const syms = this.symbols;

    for (let i = seqLen; i < this.dates.length - horizon; i++) {
      let valid = true;
      const window = [];

      // build input window
      for (let j = i - seqLen; j < i; j++) {
        const d = this.dates[j];
        const step = [];
        for (const s of syms) {
          const r = this.data[s][d];
          if (!r) { valid = false; break; }
          const vals = [r.Open, r.High, r.Low, r.Close];
          if (vals.some(v => Number.isNaN(v))) { valid = false; break; }
          step.push(...vals);
        }
        if (!valid) break;
        window.push(step);
      }
      if (!valid) continue;

      // build output labels
      const target = [];
      for (let h = 1; h <= horizon; h++) {
        const fd = this.dates[i + h];
        for (const s of syms) {
          const cur = this.data[s][this.dates[i]];
          const fut = this.data[s][fd];
          if (!cur || !fut) { valid = false; break; }
          target.push(fut.Close > cur.Close ? 1 : 0);
        }
      }
      if (!valid) continue;

      Xraw.push(window);
      Y.push(target);
    }

    if (Xraw.length === 0) throw new Error("No valid sequences (data not aligned across symbols).");

    // default split 70/15/15, but fall back if val would be zero
    const n = Xraw.length;
    let nTrain = Math.floor(n * 0.70);
    let nVal   = Math.floor(n * 0.85);
    if (nVal === nTrain) nVal = Math.min(n, nTrain + Math.max(1, Math.floor(n * 0.1)));
    if (nVal >= n) { nTrain = Math.max(1, Math.floor(n * 0.8)); nVal = nTrain; }

    const Xtr = Xraw.slice(0, nTrain);
    const Xv  = Xraw.slice(nTrain, nVal);
    const Xte = Xraw.slice(nVal);

    // compute train-only min/max per stock
    const minMax = {};
    syms.forEach((s, si) => {
      let mn = Infinity, mx = -Infinity;
      for (const w of Xtr) {
        for (const st of w) {
          const base = si * 4;
          for (let k = 0; k < 4; k++) {
            const v = st[base + k];
            if (!Number.isNaN(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
          }
        }
      }
      if (!isFinite(mn) || !isFinite(mx) || mn === mx) { mn = 0; mx = 1; }
      minMax[s] = { min: mn, max: mx };
    });

    const scale = (windows) =>
      windows.map((w) =>
        w.map((st) => {
          const out = [];
          syms.forEach((s, si) => {
            const { min, max } = minMax[s];
            const denom = Math.max(1e-9, max - min);
            const base  = si * 4;
            for (let k = 0; k < 4; k++) out.push((st[base + k] - min) / denom);
          });
          return out;
        })
      );

    const X_train = tf.tensor3d(scale(Xtr));
    const y_train = tf.tensor2d(Y.slice(0, nTrain));
    const X_val   = tf.tensor3d(scale(Xv));
    const y_val   = tf.tensor2d(Y.slice(nTrain, nVal));
    const X_test  = tf.tensor3d(scale(Xte));
    const y_test  = tf.tensor2d(Y.slice(nVal));

    console.log(`Sequences → train:${X_train.shape[0]}  val:${X_val.shape[0]}  test:${X_test.shape[0]}`);

    return { X_train, y_train, X_val, y_val, X_test, y_test, symbols: syms };
  }
}
