// app.js
// Main controller: loads data, trains RNN model, visualizes results.

import { DataLoader } from './data-loader.js';
import { RNNModel } from './rnn.js';

class StockApp {
  constructor() {
    this.loader = new DataLoader();
    this.model = null;
    this.evaluation = null;
    this.initUI();
  }

  initUI() {
    document.getElementById('csvFile').addEventListener('change', (e) => this.handleCSV(e));
    document.getElementById('trainBtn').addEventListener('click', () => this.train());
    document.getElementById('predictBtn').addEventListener('click', () => this.predict());
  }

  async handleCSV(e) {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('status').textContent = 'Loading CSV...';
    await this.loader.loadCSV(file);
    document.getElementById('status').textContent = 'Preparing sequences...';
    const data = this.loader.createSequences(30, 3);
    this.data = data;
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('status').textContent = 'Ready. Click "Train Model"';
  }

  async train() {
    const { X_train, y_train, X_test, y_test, symbols } = this.data;
    this.model = new RNNModel([30, 4 * symbols.length], 3 * symbols.length);
    document.getElementById('status').textContent = 'Training... please wait';
    await this.model.train(X_train, y_train, X_test, y_test, 25, 32);
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('status').textContent = 'Training complete. Click "Run Prediction".';
  }

  async predict() {
    const { X_test, y_test, symbols } = this.data;
    document.getElementById('status').textContent = 'Predicting...';
    const preds = this.model.predict(X_test);
    const evals = this.model.evaluatePerStock(y_test, preds, symbols);
    this.evaluation = evals;
    this.renderCharts(evals);
    document.getElementById('status').textContent = 'Done.';
    preds.dispose();
  }

  renderCharts(evals) {
    const accuracies = Object.entries(evals.stockAccuracies).sort((a, b) => b[1] - a[1]);
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: accuracies.map(a => a[0]),
        datasets: [{
          label: 'Accuracy (%)',
          data: accuracies.map(a => a[1] * 100),
          backgroundColor: accuracies.map(a => a[1] > 0.6 ? 'rgba(75,192,192,0.8)' : 'rgba(255,99,132,0.8)')
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
      div.innerHTML = `<h4>${sym} Timeline</h4><canvas id="tl-${sym}"></canvas>`;
      container.appendChild(div);
      const ctx = document.getElementById(`tl-${sym}`).getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: preds.slice(0, 50).map((_, i) => i + 1),
          datasets: [{
            label: 'Correct(1)/Wrong(0)',
            data: preds.slice(0, 50).map(p => p.correct ? 1 : 0),
            borderColor: 'rgb(75,192,192)',
            backgroundColor: 'rgba(75,192,192,0.2)',
            tension: 0.4
          }]
        },
        options: { scales: { y: { min: 0, max: 1 } } }
      });
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new StockApp());
