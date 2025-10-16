// app.js
import { loadCSV, prepareTensors } from './data-loader.js';
import { createCNNGRUModel, trainModel, evaluateModel } from './cnn_gru.js';

let tensors = null;
let model = null;
let symbols = null;

const fileInput = document.getElementById('csvFile');
const trainBtn = document.getElementById('trainBtn');
const predictBtn = document.getElementById('predictBtn');
const statusEl = document.getElementById('status');
const progressEl = document.getElementById('trainingProgress');
const accCanvas = document.getElementById('accuracyChart');
const timelineContainer = document.getElementById('timelineContainer');
let accChart = null;

fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    statusEl.textContent = 'Loading CSV...';
    const { data, symbols: syms } = await loadCSV(file);
    tensors = prepareTensors(data, syms);
    symbols = tensors.symbols;
    statusEl.textContent = `Loaded ${symbols.length} stocks, ready to train.`;
    trainBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});

trainBtn.addEventListener('click', async () => {
  if (!tensors) return;
  trainBtn.disabled = true;
  predictBtn.disabled = true;
  progressEl.value = 0;
  statusEl.textContent = 'Initializing model...';
  await tf.setBackend('webgl').catch(()=>tf.setBackend('cpu'));
  await tf.ready();
  model = createCNNGRUModel(symbols.length);
  statusEl.textContent = 'Training started...';

  await trainModel(model, tensors.X_train, tensors.y_train, tensors.X_val, tensors.y_val,
    (p)=>progressEl.value = p,
    (msg)=>statusEl.textContent = msg
  );

  statusEl.textContent = 'Training complete.';
  predictBtn.disabled = false;
});

predictBtn.addEventListener('click', async () => {
  if (!model || !tensors) return;
  statusEl.textContent = 'Running predictions...';
  const result = await evaluateModel(model, tensors.X_test, tensors.y_test, symbols);
  renderAccuracyChart(result.stockAccuracies);
  renderTimelines(result.stockAccuracies, result.stockPredictions);
  statusEl.textContent = 'Prediction complete.';
  tensors.X_train.dispose();
  tensors.y_train.dispose();
  tensors.X_val.dispose();
  tensors.y_val.dispose();
  tensors.X_test.dispose();
  tensors.y_test.dispose();
});

function renderAccuracyChart(stockAccuracies) {
  const sorted = Object.entries(stockAccuracies).sort((a,b)=>b[1]-a[1]);
  const labels = sorted.map(x=>x[0]);
  const values = sorted.map(x=>x[1]);
  if (accChart) accChart.destroy();
  accChart = new Chart(accCanvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Accuracy',
        data: values,
        backgroundColor: '#4caf50'
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          max: 1
        }
      },
      plugins: {
        legend: {display:false}
      }
    }
  });
}

function renderTimelines(stockAccuracies, stockPredictions) {
  timelineContainer.innerHTML = '';
  const sorted = Object.entries(stockAccuracies).sort((a,b)=>b[1]-a[1]).slice(0,3);
  for (const [sym] of sorted) {
    const canvas = document.createElement('canvas');
    timelineContainer.appendChild(canvas);
    const data = stockPredictions[sym];
    const labels = data.map((_,i)=>i);
    const correct = data.map(d=>d.correct?1:0);
    new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: `${sym} Correct(1)/Wrong(0)`,
          data: correct,
          borderColor: '#2196f3',
          fill: false
        }]
      },
      options: {
        scales: { y: {min:0, max:1, ticks:{stepSize:1}} }
      }
    });
  }
}
