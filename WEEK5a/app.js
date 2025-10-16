// app.js
// Controls data loading, training, and visualization

import { DataLoader } from './data-loader.js';
import { RNN_GRU_Model } from './rnn_gru.js';

class StockApp {
  constructor() {
    this.loader = new DataLoader();
    this.model = null;
    this.data = null;
    this.initUI();
  }

  initUI() {
    document.getElementById('csvFile').addEventListener('change', e => this.loadFile(e));
    document.getElementById('trainBtn').addEventListener('click', () => this.train());
    document.getElementById('predictBtn').addEventListener('click', () => this.predict());
  }

  async loadFile(e) {
    const f = e.target.files[0];
    if (!f) return;
    const s = document.getElementById('status');
    try {
      s.textContent = 'Loading CSV...';
      await this.loader.loadCSV(f);
      s.textContent = 'Creating 30-day sequences...';
      this.data = this.loader.createSequences(30, 3);
      document.getElementById('trainBtn').disabled = false;
      s.textContent = 'Data ready. Click Train.';
    } catch (err) {
      console.error(err); s.textContent = `Error: ${err.message}`;
    }
  }

  async train() {
    const { X_train, y_train, X_val, y_val, X_test, y_test, symbols } = this.data;
    console.log("Starting training with:", X_train.shape, y_train.shape);

    this.model = new RNN_GRU_Model([30, 4 * symbols.length], 3 * symbols.length);
    document.getElementById('status').textContent = 'Training model...';

    await this.model.train(X_train, y_train, X_val, y_val, 50, 64);

    document.getElementById('predictBtn').disabled = false;
    document.getElementById('status').textContent = 'Training complete. Run Prediction.';
  }

  async predict() {
    const { X_test, y_test, symbols } = this.data;
    const s = document.getElementById('status');
    s.textContent = 'Predicting...';
    const preds = this.model.predict(X_test);
    const evals = this.model.evaluate(y_test, preds, symbols);
    preds.dispose();
    this.render(evals);
    s.textContent = 'Prediction complete.';
  }

  render(evals) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    const entries = Object.entries(evals.stockAccuracies).sort((a, b) => b[1] - a[1]);
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: entries.map(e => e[0]),
        datasets: [{
          label: 'Accuracy (%)',
          data: entries.map(e => e[1] * 100),
          backgroundColor: entries.map(e => e[1] > 0.6 ? 'rgba(75,192,192,0.8)' : 'rgba(255,99,132,0.8)')
        }]
      },
      options: { indexAxis: 'y', scales: { x: { beginAtZero: true, max: 100 } } }
    });

    const cont = document.getElementById('timelineContainer');
    cont.innerHTML = '';
    Object.entries(evals.stockPredictions).slice(0, 3).forEach(([sym, preds]) => {
      const div = document.createElement('div');
      div.className = 'stock-chart';
      div.innerHTML = `<h4>${sym}</h4><canvas id="tl-${sym}"></canvas>`;
      cont.appendChild(div);
      const c = document.getElementById(`tl-${sym}`).getContext('2d');
      new Chart(c, {
        type: 'line',
        data: {
          labels: preds.slice(0, 50).map((_, i) => i + 1),
          datasets: [{
            label: 'Correct(1)/Wrong(0)',
            data: preds.slice(0, 50).map(p => p.correct ? 1 : 0),
            borderColor: 'rgb(75,192,192)',
            backgroundColor: 'rgba(75,192,192,0.2)',
            tension: 0.3
          }]
        },
        options: { scales: { y: { min: 0, max: 1 } } }
      });
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new StockApp());
