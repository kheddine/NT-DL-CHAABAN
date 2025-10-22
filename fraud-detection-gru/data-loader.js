// data-loader.js
// Handles reading local CSV and preparing tensors for GRU model
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.16.0/dist/tf.min.js";

export class DataLoader {
  constructor() {
    this.X_train = null;
    this.y_train = null;
    this.X_test = null;
    this.y_test = null;
  }

  async loadCSV(file) {
    const text = await file.text();
    const rows = text.trim().split("\n").map(r => r.split(","));
    const header = rows.shift();
    const stepIdx = header.indexOf("step");
    const amountIdx = header.indexOf("amount");
    const oldBalIdx = header.indexOf("oldbalanceOrg");
    const newBalIdx = header.indexOf("newbalanceOrig");
    const destOldIdx = header.indexOf("oldbalanceDest");
    const destNewIdx = header.indexOf("newbalanceDest");
    const fraudIdx = header.indexOf("isFraud");

    // parse numeric rows
    const data = rows.map(r => [
      +r[stepIdx], +r[amountIdx], +r[oldBalIdx], +r[newBalIdx],
      +r[destOldIdx], +r[destNewIdx], +r[fraudIdx]
    ]);

    // normalize each column except target
    const features = data.map(r => r.slice(0, -1));
    const labels = data.map(r => r[r.length - 1]);
    const fTensor = tf.tensor2d(features);
    const { mean, variance } = tf.moments(fTensor, 0);
    const normalized = fTensor.sub(mean).div(variance.sqrt());
    const nFeat = normalized.arraySync();
    const nData = nFeat.map((f, i) => [...f, labels[i]]);

    // simple sequential windows
    const seqLen = 5;
    const X = [];
    const y = [];
    for (let i = 0; i < nData.length - seqLen; i++) {
      const seq = nData.slice(i, i + seqLen);
      const label = nData[i + seqLen][nData[i + seqLen].length - 1];
      X.push(seq.map(r => r.slice(0, -1)));
      y.push([label]);
    }

    const split = Math.floor(X.length * 0.8);
    this.X_train = tf.tensor3d(X.slice(0, split));
    this.y_train = tf.tensor2d(y.slice(0, split));
    this.X_test = tf.tensor3d(X.slice(split));
    this.y_test = tf.tensor2d(y.slice(split));
  }
}
