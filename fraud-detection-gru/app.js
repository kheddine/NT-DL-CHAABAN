// app.js
// Main UI logic tying data loader and GRU model together
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.16.0/dist/tf.min.js";
import { DataLoader } from "./data-loader.js";
import { GRUFraudModel } from "./module.js";

const fileInput = document.getElementById("fileInput");
const trainBtn = document.getElementById("trainBtn");
const predictBtn = document.getElementById("predictBtn");
const statusDiv = document.getElementById("status");
const chartEl = document.getElementById("chart");

let loader = new DataLoader();
let model = null;

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  statusDiv.innerText = "Loading CSV...";
  await loader.loadCSV(file);
  statusDiv.innerText = `Data loaded. Train: ${loader.X_train.shape[0]} samples.`;
});

trainBtn.addEventListener("click", async () => {
  if (!loader.X_train) return alert("Load data first!");
  model = new GRUFraudModel([loader.X_train.shape[1], loader.X_train.shape[2]]);
  statusDiv.innerText = "Training...";
  await model.train(loader.X_train, loader.y_train, loader.X_test, loader.y_test,
    (epoch, logs) => {
      statusDiv.innerText = `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc * 100).toFixed(2)}%`;
    });
  const evalRes = await model.evaluate(loader.X_test, loader.y_test);
  statusDiv.innerText = `Done. Test acc: ${(evalRes.acc * 100).toFixed(2)}%`;
});

predictBtn.addEventListener("click", async () => {
  if (!model) return alert("Train model first!");
  const preds = model.predict(loader.X_test).dataSync();
  const labels = loader.y_test.dataSync();
  const results = Array.from(preds).map((p, i) => ({ p, y: labels[i] }));

  // simple visualization
  const ctx = chartEl.getContext("2d");
  ctx.clearRect(0, 0, chartEl.width, chartEl.height);
  results.slice(0, 200).forEach((r, i) => {
    ctx.fillStyle = (r.y === 1 && r.p > 0.5) || (r.y === 0 && r.p <= 0.5) ? "green" : "red";
    ctx.fillRect(i * 4, chartEl.height - r.p * chartEl.height, 3, r.p * chartEl.height);
  });
  statusDiv.innerText += " | Predictions visualized (green=correct, red=wrong)";
});
