// app.js
// App controller: load CSV → build tensors → train → predict → visualize

import { DataLoader } from './data-loader.js';
import { RNN_GRU_Model } from './rnn_gru.js';

class StockApp {
  constructor() {
    this.loader = new DataLoader();
    this.model  = null;
    this.data   = null;
    this.chart  = null;
    this.initUI();
  }

  initUI() {
    document.getElementById('csvFile').addEventListener('change', (e) => this.onCSV(e));
    document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
    document.getElementById('predictBtn').addEventListener('click', () => this.onPredict());
  }

  setStatus(msg) {
    const s = document.getElementById('status');
    if (s) s.textContent = msg;
    console.log('[STATUS]', msg);
  }

  async onCSV(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      this.setStatus('Reading CSV…');
      await this.loader.loadCSV(file);
      this.setStatus('Creating 30-day sequences…');
      this.data = this.loader.createSequences(30, 3);
      this.setStatus('Data ready. Click "Train Model".');
      document.getElementById('trainBtn').disabled = false;
    } catch (err) {
      console.error(err);
      this.setStatus(`Error: ${err.message}`);
    }
  }

  async onTrain() {
    if (!this.data) { this.setStatus('Load data first.'); return; }
    const { X_train, y_train, X_val, y_val, symbols } = this.data;

    // Log shapes for sanity
    console.log('X_train:', X_train.shape, 'y_train:', y_train.shape);
    console.log('X_val  :', X_val.shape,   'y_val  :', y_val.shape);

    try {
      this.setStatus('Initializing & training model…');
      this.model = new RNN_GRU_Model([30, 4 * symbols.length], 3 * symbols.length);
      await this.model.train(X_train, y_train, X_val, y_val, 50, 64);
      document.getElementById('predictBtn').disabled = false;
      this.setStatus('Training complete. Click "Run Prediction".');
    } catch (err) {
      console.error(err);
      this.setStatus(`Training error: ${err.message}`);
    }
  }

  async onPredict() {
    if (!this.model) { this.setStatus('Train the model first.'); return; }
    const { X_test, y_test, symbols } = this.data;

    try {
      this.setStatus('Running predictions…');
      const preds = this.model.predict(X_test);
      const evals = this.model.evaluate(y_test, preds, symbols);
      if (preds?.dispose) preds.dispose();
      this.render(evals);
      this.setStatus('Prediction complete.');
    } catch (err) {
      console.error(err);
      this.setStatus(`Prediction error: ${err.message}`);
    }
  }

  render(evals) {
    // Accuracy bar (sorted)
    const accEntries = Object.entries(evals.stockAccuracies).sort((a, b) => b[1] - a[1]);
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    if (this.chart) this.chart.destroy();
    this.chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: accEntries.map(([s]) => s),
        datasets: [{
          label: 'Accuracy (%)',
          data: accEntries.map(([, v]) => v * 100),
          backgroundColor: accEntries.map(([, v]) => v >= 0.6 ? '#6be4c1bb' : '#ff6b6bbb')
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        scales: { x: { beginAtZero: true, max: 100, grid: { color: '#223' } }, y: { grid: { color: '#223' } } }
      }
    });

    // Timelines (top 3)
    const container = document.getElementById('timelineContainer');
    container.innerHTML = '';
    Object.entries(evals.stockPredictions).slice(0, 3).forEach(([sym, preds]) => {
      const wrap = document.createElement('div');
      wrap.className = 'stock-chart';
      wrap.innerHTML = `<h4 style="margin:6px 0">${sym}</h4><canvas id="tl-${sym}"></canvas>`;
      container.appendChild(wrap);
      const c = document.getElementById(`tl-${sym}`).getContext('2d');
      new Chart(c, {
        type: 'line',
        data: {
          labels: preds.slice(0, 50).map((_, i) => i + 1),
          datasets: [{
            label: 'Correct(1)/Wrong(0)',
            data: preds.slice(0, 50).map(p => p.correct ? 1 : 0),
            borderColor: '#6be4c1',
            backgroundColor: '#6be4c120',
            tension: 0.3,
            pointRadius: 2
          }]
        },
        options: { scales: { y: { min: 0, max: 1 } } }
      });
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new StockApp());
