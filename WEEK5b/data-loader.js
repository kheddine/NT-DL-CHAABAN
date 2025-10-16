// data-loader.js
// Build [samples, 12, 20] inputs (Open+Close for 10 stocks) and 3-day-ahead 30-bit outputs.
// Train-only min-max normalization. Robust splits with fallback.

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
    const required = ["Date", "Symbol", "Open", "Close"];
    for (const h of required) {
      if (!headers.includes(h)) {
        throw new Error(`Missing column "${h}". Found: ${headers.join(", ")}`);
      }
    }
    const idx = Object.fromEntries(headers.map((h, i) => [h, i]));

    const data = {};
    const syms = new Set();
    const dts = new Set();

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(",");
      if (parts.length < headers.length) continue;
      const date = parts[idx.Date]?.trim();
      const symbol = parts[idx.Symbol]?.trim();
      const open = parseFloat(parts[idx.Open]);
      const close = parseFloat(parts[idx.Close]);
      if (!date || !symbol) continue;
      if ([open, close].some(v => Number.isNaN(v))) continue;

      if (!data[symbol]) data[symbol] = {};
      data[symbol][date] = { Open: open, Close: close };
      syms.add(symbol);
      dts.add(date);
    }

    this.symbols = Array.from(syms).sort().slice(0, 10);
    this.dates = Array.from(dts).sort();
    this.data = data;

    if (this.symbols.length === 0) throw new Error("No valid symbols parsed from CSV.");
    if (this.dates.length < 20) throw new Error("Not enough dates for 12-day windows.");
  }

  createSequences(seqLen = 12, horizon = 3) {
    const Xraw = [];
    const Y = [];
    const syms = this.symbols;

    for (let i = seqLen; i < this.dates.length - horizon; i++) {
      let valid = true;
      const windowSteps = [];

      // 12-day input window
      for (let j = i - seqLen; j < i; j++) {
        const d = this.dates[j];
        const step = [];
        for (const s of syms) {
          const rec = this.data[s][d];
          if (!rec) { valid = false; break; }
          const vals = [rec.Open, rec.Close];
          if (vals.some(v => Number.isNaN(v))) { valid = false; break; }
          step.push(...vals);
        }
        if (!valid) break;
        windowSteps.push(step);
      }
      if (!valid) continue;

      // 3-day-ahead binary labels per stock
      const target = [];
      for (let h = 1; h <= horizon; h++) {
        const fd = this.dates[i + h];
        for (const s of syms) {
          const now = this.data[s][this.dates[i]];
          const fut = this.data[s][fd];
          if (!now || !fut) { valid = false; break; }
          target.push(fut.Close > now.Close ? 1 : 0);
        }
      }
      if (!valid) continue;

      Xraw.push(windowSteps);
      Y.push(target);
    }

    if (!Xraw.length) throw new Error("No valid sequences (symbols may not align by date).");

    // 70/15/15 with fallback to 80/20 if needed
    const n = Xraw.length;
    let nTrain = Math.floor(n * 0.70);
    let nVal = Math.floor(n * 0.85);
    if (nVal === nTrain) nVal = Math.min(n, nTrain + Math.max(1, Math.floor(n * 0.1)));
    if (nVal >= n) { nTrain = Math.max(1, Math.floor(n * 0.8)); nVal = nTrain; }

    const Xtr = Xraw.slice(0, nTrain);
    const Xv  = Xraw.slice(nTrain, nVal);
    const Xte = Xraw.slice(nVal);

    // Train-only min/max per stock (on Open/Close)
    const minMax = {};
    syms.forEach((s, si) => {
      let mn = Infinity, mx = -Infinity;
      for (const w of Xtr) {
        for (const st of w) {
          const base = si * 2;
          const a = st[base], b = st[base + 1];
          if (!Number.isNaN(a)) { if (a < mn) mn = a; if (a > mx) mx = a; }
          if (!Number.isNaN(b)) { if (b < mn) mn = b; if (b > mx) mx = b; }
        }
      }
      if (!isFinite(mn) || !isFinite(mx) || mn === mx) { mn = 0; mx = 1; }
      minMax[s] = { min: mn, max: mx };
    });

    const scale = arr => arr.map(w => w.map(st => {
      const out = [];
      syms.forEach((s, si) => {
        const { min, max } = minMax[s];
        const denom = Math.max(1e-9, max - min);
        const base = si * 2;
        out.push((st[base] - min) / denom);
        out.push((st[base + 1] - min) / denom);
      });
      return out;
    }));

    const X_train = tf.tensor3d(scale(Xtr));      // [N, 12, 20]
    const y_train = tf.tensor2d(Y.slice(0, nTrain)); // [N, 30]
    const X_val   = tf.tensor3d(scale(Xv));
    const y_val   = tf.tensor2d(Y.slice(nTrain, nVal));
    const X_test  = tf.tensor3d(scale(Xte));
    const y_test  = tf.tensor2d(Y.slice(nVal));

    console.log("Tensors:", {
      X_train: X_train.shape, y_train: y_train.shape,
      X_val: X_val.shape, y_val: y_val.shape,
      X_test: X_test.shape, y_test: y_test.shape
    });

    return { X_train, y_train, X_val, y_val, X_test, y_test, symbols: syms };
  }
}
