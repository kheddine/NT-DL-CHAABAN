/**
 * app.js
 * Coordinates UI interactions, dataset preparation, model training, and
 * visualisations for the client-side stock prediction demo.
 */

import { loadCSVFile } from './data-loader.js';
import { ensureBackend, createRNNGRUModel, trainModel, evaluateModel } from './rnn_gru.js';

const elements = {
  fileInput: document.getElementById('csvFile'),
  trainBtn: document.getElementById('trainBtn'),
  predictBtn: document.getElementById('predictBtn'),
  status: document.getElementById('status'),
  progress: document.getElementById('trainingProgress'),
  accuracyCanvas: document.getElementById('accuracyChart'),
  timelineCanvas: document.getElementById('timelineChart'),
};

let dataset = null;
let model = null;
let charts = {
  accuracy: null,
  timeline: null,
};
let latestEvaluation = null;

function setStatus(message, type = 'info') {
  elements.status.textContent = message;
  elements.status.style.color = type === 'error' ? '#ff6b6b' : '#a0c4ff';
}

function setProgress(value) {
  elements.progress.value = value;
}

function resetCharts() {
  if (charts.accuracy) {
    charts.accuracy.destroy();
    charts.accuracy = null;
  }
  if (charts.timeline) {
    charts.timeline.destroy();
    charts.timeline = null;
  }
}

function enableControls({ train, predict, file }) {
  elements.trainBtn.disabled = !train;
  elements.predictBtn.disabled = !predict;
  elements.fileInput.disabled = !file;
}

async function onCSVChange(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    setStatus('Loading and processing CSV data...');
    enableControls({ train: false, predict: false, file: false });
    resetCharts();

    if (dataset) {
      dataset.X_train?.dispose();
      dataset.y_train?.dispose();
      dataset.X_val?.dispose();
      dataset.y_val?.dispose();
      dataset.X_test?.dispose();
      dataset.y_test?.dispose();
    }
    dataset = null;

    const tensors = await loadCSVFile(file, {
      onStatus: (msg) => setStatus(msg),
    });
    dataset = tensors;

    setStatus('Dataset ready. Click "Train Model" to begin training.');
    setProgress(0);
    enableControls({ train: true, predict: false, file: true });
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Failed to load CSV data.', 'error');
    enableControls({ train: false, predict: false, file: true });
  }
}

async function onTrainClick() {
  if (!dataset) {
    setStatus('Please load a CSV file before training.', 'error');
    return;
  }

  try {
    enableControls({ train: false, predict: false, file: false });
    setStatus('Initialising TensorFlow backend...');
    await ensureBackend((msg) => setStatus(msg));

    if (model) {
      model.dispose();
    }
    model = createRNNGRUModel(dataset.symbols.length);

    setStatus('Training model...');
    setProgress(0);

    await trainModel(
      model,
      dataset,
      {
        onEpochBegin: (epoch) => {
          setStatus(`Epoch ${epoch + 1} / 50 in progress...`);
        },
        onEpochEnd: (epoch, logs) => {
          const progress = (epoch + 1) / 50;
          setProgress(progress);
          const acc = logs?.binaryAccuracy || logs?.val_binaryAccuracy || 0;
          setStatus(
            `Epoch ${epoch + 1} complete. Binary accuracy: ${(acc * 100).toFixed(2)}%`
          );
        },
      }
    );

    setProgress(1);
    setStatus('Training complete. Click "Generate Predictions" to evaluate.');
    enableControls({ train: true, predict: true, file: true });
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Training failed.', 'error');
    enableControls({ train: true, predict: false, file: true });
  }
}

function buildAccuracyChart(stockAccuracies) {
  const labels = Object.keys(stockAccuracies);
  const data = labels.map((label) => Number((stockAccuracies[label] * 100).toFixed(2)));

  charts.accuracy = new Chart(elements.accuracyCanvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Accuracy (%)',
          data,
          backgroundColor: '#3aa0ff',
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: {
            callback: (value) => `${value}%`,
            color: '#ccc',
          },
          grid: {
            color: '#333',
          },
        },
        y: {
          ticks: {
            color: '#ccc',
          },
          grid: {
            display: false,
          },
        },
      },
      plugins: {
        legend: {
          labels: { color: '#eee' },
        },
      },
    },
  });
}

function buildTimelineChart(stockPredictions) {
  const sortedStocks = Object.entries(latestEvaluation.stockAccuracies)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([symbol]) => symbol);

  const maxLength = 50;
  const labels = Array.from({ length: maxLength }, (_, i) => i + 1);
  const datasets = sortedStocks.map((symbol, index) => {
    const colorPalette = ['#8d6bff', '#3aa0ff', '#ff6b6b'];
    const entries = stockPredictions[symbol] || [];
    const data = new Array(maxLength).fill(null);
    for (let i = 0; i < Math.min(entries.length, maxLength); i++) {
      data[i] = entries[i].correct;
    }
    return {
      label: `${symbol} (1=correct)`,
      data,
      fill: false,
      tension: 0,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      borderColor: colorPalette[index % colorPalette.length],
      pointBackgroundColor: colorPalette[index % colorPalette.length],
    };
  });

  charts.timeline = new Chart(elements.timelineCanvas.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets,
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: {
            stepSize: 1,
            callback: (value) => (value === 1 ? 'Correct' : 'Incorrect'),
            color: '#ccc',
          },
          grid: {
            color: '#333',
          },
        },
        x: {
          ticks: {
            color: '#ccc',
          },
          grid: {
            color: '#2a2a2a',
          },
        },
      },
      plugins: {
        legend: {
          labels: { color: '#eee' },
        },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              ctx.raw === null
                ? `${ctx.dataset.label}: No prediction`
                : `${ctx.dataset.label}: ${ctx.raw === 1 ? 'Correct' : 'Incorrect'} (Truth ${
                    stockPredictions[ctx.dataset.label.split(' ')[0]][ctx.dataIndex]?.truth
                  }, Pred ${
                    stockPredictions[ctx.dataset.label.split(' ')[0]][ctx.dataIndex]?.pred
                  })`,
          },
        },
      },
    },
  });
}

async function onPredictClick() {
  if (!dataset || !model) {
    setStatus('Load data and train the model before generating predictions.', 'error');
    return;
  }

  try {
    enableControls({ train: false, predict: false, file: false });
    setStatus('Running inference on test set...');

    latestEvaluation = await evaluateModel(model, dataset);
    const { stockAccuracies, stockPredictions } = latestEvaluation;

    setStatus('Predictions ready. Rendering charts.');
    resetCharts();
    buildAccuracyChart(stockAccuracies);
    buildTimelineChart(stockPredictions);

    setStatus('Evaluation complete. Charts updated.');
    enableControls({ train: true, predict: true, file: true });
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Prediction failed.', 'error');
    enableControls({ train: true, predict: true, file: true });
  }
}

elements.fileInput.addEventListener('change', onCSVChange);
elements.trainBtn.addEventListener('click', onTrainClick);
elements.predictBtn.addEventListener('click', onPredictClick);

setStatus('Load a CSV file to begin.');
setProgress(0);
