// app.js
// Connects UI to data loader and model, handles training and visualizations.

import { DataLoader } from './data-loader.js';
import { StockRNN } from './rnn.js';

const fileInput = document.getElementById('fileInput');
const trainBtn = document.getElementById('trainBtn');
const logDiv = document.getElementById('log');
const chartDiv = document.getElementById('chartContainer');

let loader, model, data;

fileInput.addEventListener('change', async (e) => {
  loader = new DataLoader();
  const file = e.target.files[0];
  log('Loading and preparing data...');
  data = await loader.prepareData(file);
  log(`Loaded ${data.symbols.length} stocks, ${data.X_train.shape[0]} training samples.`);
  trainBtn.disabled = false;
});

trainBtn.addEventListener('click', async () => {
  trainBtn.disabled = true;
  model = new StockRNN([30, 4 * data.symbols.length], 3 * data.symbols.length);
  log('Training model...');
  await model.train(data.X_train, data.y_train, 15, 32, (epoch, logs) => {
    log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.binaryAccuracy * 100).toFixed(2)}%`);
  });

  log('Evaluating model...');
  const results = await model.evaluate(data.X_test, data.y_test);
  log(`Test Accuracy: ${(results.accuracy * 100).toFixed(2)}%`);

  const preds = model.predict(data.X_test).arraySync();
  visualizeAccuracy(preds, data.y_test.arraySync(), data.symbols);
});

function log(msg) {
  logDiv.textContent += msg + '\n';
}

function visualizeAccuracy(preds, truths, symbols) {
  const accuracies = symbols.map((sym, i) => {
    let correct = 0, total = 0;
    for (let s = 0; s < preds.length; s++) {
      for (let h = 0; h < 3; h++) {
        const idx = i * 3 + h;
        const p = preds[s][idx] > 0.5 ? 1 : 0;
        if (p === truths[s][idx]) correct++;
        total++;
      }
    }
    return { sym, acc: correct / total };
  }).sort((a, b) => b.acc - a.acc);

  const ctx = document.createElement('canvas');
  chartDiv.innerHTML = '';
  chartDiv.appendChild(ctx);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: accuracies.map(a => a.sym),
      datasets: [{
        label: 'Accuracy',
        data: accuracies.map(a => a.acc * 100),
      }],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 100 } },
    },
  });
}
