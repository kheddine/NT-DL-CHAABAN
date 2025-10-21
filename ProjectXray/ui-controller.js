/**
 * ui-controller.js
 * Handles user interaction, visualization, and safety messaging.
 */

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const trainingSection = document.getElementById('training-section');
const inferenceSection = document.getElementById('inference-section');
const trainModeBtn = document.getElementById('train-mode-btn');
const inferenceModeBtn = document.getElementById('inference-mode-btn');
const startTrainingBtn = document.getElementById('start-training');
const runInferenceBtn = document.getElementById('run-inference');
const outputDiv = document.getElementById('output');
const trainingStatus = document.getElementById('training-status');

let mode = null;
let uploadedFiles = [];

/** Drag and drop handlers */
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', e => handleFiles(e.target.files));

/** Switch modes */
trainModeBtn.onclick = () => switchMode('train');
inferenceModeBtn.onclick = () => switchMode('inference');

function switchMode(selected) {
  mode = selected;
  trainingSection.style.display = selected === 'train' ? 'block' : 'none';
  inferenceSection.style.display = selected === 'inference' ? 'block' : 'none';
  outputDiv.innerHTML = '';
  trainingStatus.innerHTML = '';
}

/** Display uploaded images */
function handleFiles(files) {
  uploadedFiles = Array.from(files);
  imagePreview.innerHTML = '';
  uploadedFiles.forEach(file => {
    const reader = new FileReader();
    reader.onload = e => {
      const div = document.createElement('div');
      div.className = 'preview-item';
      div.innerHTML = `<img src="${e.target.result}" alt="Uploaded X-ray"/>`;
      imagePreview.appendChild(div);
    };
    reader.readAsDataURL(file);
  });
}

/** Start training process */
startTrainingBtn.onclick = async () => {
  try {
    if (uploadedFiles.length === 0) return alert('Please upload labeled images first.');
    trainingStatus.textContent = 'Preparing data...';
    const labels = uploadedFiles.map(f => f.name.toLowerCase().includes('covid') ? 1 : 0);
    const { xs, ys } = await createDataset(uploadedFiles, labels);

    trainingStatus.textContent = 'Building model...';
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const lr = parseFloat(document.getElementById('lr').value);
    await buildBaseModel(false, lr);

    trainingStatus.textContent = 'Training...';
    await trainModel(xs, ys, xs, ys, epochs, batchSize);

    const evalRes = await evaluateModel(xs, ys);
    trainingStatus.innerHTML = `
      <p>Training complete.</p>
      <p><strong>Accuracy:</strong> ${(evalRes.accuracy * 100).toFixed(2)}%</p>
      <p><strong>Loss:</strong> ${evalRes.loss.toFixed(4)}</p>
    `;
    xs.dispose(); ys.dispose();
  } catch (err) {
    console.error(err);
    alert('Error during training: ' + err.message);
  }
};

/** Run inference on uploaded images */
runInferenceBtn.onclick = async () => {
  try {
    if (uploadedFiles.length === 0) return alert('Please upload images first.');
    if (!covidModel) await buildBaseModel(false);

    outputDiv.innerHTML = '';
    for (const file of uploadedFiles) {
      const imgTensor = await preprocessImage(file);
      const batched = imgTensor.expandDims(0);
      const probs = await predictImages(batched);
      const [normal, covid] = probs[0];
      const label = covid > normal ? 'COVID-19' : 'Normal';
      const confidence = (Math.max(normal, covid) * 100).toFixed(2);

      const warnColor = covid > 0.7 ? 'red' : covid > 0.5 ? 'orange' : 'green';
      const div = document.createElement('div');
      div.className = 'prediction';
      div.innerHTML = `
        <p><strong>${file.name}</strong></p>
        <p>Prediction: <span style="color:${warnColor}">${label}</span></p>
        <p>Confidence: ${confidence}%</p>
        <small>Educational result only – not for diagnosis.</small>
      `;
      outputDiv.appendChild(div);
      tf.dispose([imgTensor, batched]);
    }
  } catch (err) {
    console.error(err);
    alert('Error during inference: ' + err.message);
  }
};

/** Accessibility enhancements */
uploadArea.addEventListener('keypress', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
