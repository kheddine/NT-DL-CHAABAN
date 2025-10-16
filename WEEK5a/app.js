// app.js
// Wire up: load CSV -> tensors -> train CNN+GRU -> predict -> charts

import { DataLoader } from './data-loader.js';
import { CNN_GRU_Model } from './cnn_gru.js';

class App {
  constructor() {
    this.loader = new DataLoader();
    this.model = null;
    this.data = null;
    this.accChart = null;
    this.init();
  }

  $(id) { return document.getElementById(id); }
  setStatus(msg) {
    const el = this.$('status'); if (el) el.textContent = msg;
    console.log('[STATUS]', msg);
  }

  init() {
    this.$('csvFile').addEventListener('change', e => this.onFile(e));
    this.$('trainBtn').addEventListener('click', () => this.onTrain());
    this.$('predictBtn').addEventListener('click', () => this.onPredict());
  }

  async onFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      this.setStatus('Reading CSV…');
      await this.loader.loadCSV(file);
      this.setStatus('Building 12-day Open/Close sequences…');
      this.data = this.loader.createSequences(12, 3);
      this.$('trainBtn').disabled = false;
      this.setStatus('Data ready. Click "Train Model".');
    } catch (err) {
      console.error(err);
      this.setStatus(`Error: ${err.message}`);
    }
  }

  async onTrain() {
    if (!this.data) { this.setStatus('Load data first.'); return; }
    const { X_train, y_train, X_val, y_val, symbols } = this.data;
    try {
      this.setStatus('Training model…');
      this.model = new CNN_GRU_Model([12, 2 * symbols.length], 3 * symbols.length);
      await this.model.train(X_train, y_train, X_val, y_val, 50, 64);
      this.$('predictBtn').disabled = false;
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
    // Accuracy bar chart
    const entries = Object.entries(evals.stockAccuracies).sort((a, b) => b[1] - a[1]);
    const ctx = this.$('accuracyChart').getContext('2d');
    if (this.accChart) this.accChart.destroy();
    this.accChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: entries.map(e => e[0]),
        datasets: [{
          label: 'Accuracy (%)',
          data: entries.map(e => (e[1] * 100)),
          backgroundColor: entries.map(e => e[1] >= 0.6 ? '#6be4c1cc' : '#ff6b6bcc')
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        scales: { x: { beginAtZero: true, max: 100 } }
      }
    });

    // Timelines for top 3 stocks
    const container = this.$('timelineContainer');
    container.innerHTML = '';
    Object.entries(evals.stockPredictions).slice(0, 3).forEach(([sym, preds]) => {
      const card = document.createElement('div');
      card.className = 'stock-chart';
      card.innerHTML = `<h4 style="margin:6px 0">${sym}</h4><canvas id="tl-${sym}"></canvas>`;
      container.appendChild(card);

      const c = this.$(`tl-${sym}`).getContext('2d');
      new Chart(c, {
        type: 'line',
        data: {
          labels: preds.slice(0, 50).map((_, i) => i + 1),
          datasets: [{
            label: 'Correct (1) / Wrong (0)',
            data: preds.slice(0, 50).map(p => p.correct ? 1 : 0),
            borderColor: '#6be4c1',
            backgroundColor: '#6be4c140',
            tension: 0.3,
            pointRadius: 2
          }]
        },
        options: { scales: { y: { min: 0, max: 1 } } }
      });
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new App());
