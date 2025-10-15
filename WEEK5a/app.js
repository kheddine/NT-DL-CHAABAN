// app.js
// Main controller for file loading, training, and visualization.

import { DataLoader } from './data-loader.js';
import { RNNModel } from './rnn.js';

class StockApp {
  constructor() {
    this.loader = new DataLoader();
    this.model = null;
    this.data = null;
    this.evals = null;
    this.initUI();
  }

  initUI() {
    document.getElementById('csvFile').addEventListener('change', (e) => this.handleFile(e));
    document.getElementById('trainBtn').addEventListener('click', () => this.train());
    document.getElementById('predictBtn').addEventListener('click', () => this.predict());
  }

  async handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const status = document.getElementById('status');
    try {
      status.textContent = 'Reading CSV...';
      await this.loader.loadCSV(file);
      status.textContent = 'Preparing 30-day sequences...';
      this.data = this.loader.createSequences(30, 3);
      document.getElementById('trainBtn').disabled = false;
      status.textContent = `Loaded ${this.data.symbols.length} stocks. Ready to train.`;
    } catch (err) {
      console.error(err);
      status.textContent = `Error: ${err.message}`;
    }
  }

  async train() {
    const { X_train, y_train, X_test, y_test, symbols } = this.data;
    this.model = new RNNModel([30, 4 * symbols.length], 3 * symbols.length);
    document.getElementById('status').textContent = 'Training model...';
    await this.model.train(X_train, y_train, X_test, y_test, 25, 32);
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('status').textContent = 'Training complete. Click Run Prediction.';
  }

  async predict() {
    const { X_test, y_test, symbols } = this.data;
    document.getElementById('status').textContent = 'Running predictions...';
    const preds = this.model.predict(X_test);
    const evals = this.model.evaluate(y_test, preds, symbols);
    this.evals = evals;
    this.render(evals);
    preds.dispose();
    document.getElementById('status').textContent = 'Predictions ready.';
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
      options: {
        indexAxis: 'y',
        scales: { x: { beginAtZero: true, max: 100 } }
      }
    });

    const container = document.getElementById('timelineContainer');
    container.innerHTML = '';
    Object.entries(evals.stockPredictions).slice(0, 3).forEach(([sym, preds]) => {
      const div = document.createElement('div');
      div.className = 'stock-chart';
      div.innerHTML = `<h4>${sym} Predictions</h4><canvas id="tl-${sym}"></canvas>`;
      container.appendChild(div);
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
