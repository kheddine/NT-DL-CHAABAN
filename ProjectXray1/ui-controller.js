/**
 * UI Controller Module
 * Manages user interactions, visualizations, and application state
 * Coordinates between data loading and model operations
 */

// Import necessary modules
import { XRayClassifier } from "./model.js"
import { DataLoader } from "./data-loader.js"
import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"

// Initialize components
const classifier = new XRayClassifier()
const dataLoader = new DataLoader(224)

// State management
const appState = {
  mode: "inference",
  modelLoaded: false,
  isTraining: false,
  inferenceImages: [],
  covidImages: [],
  normalImages: [],
  trainingData: null,
}

// DOM elements
const elements = {
  // Mode switching
  modeBtns: document.querySelectorAll(".mode-btn"),
  inferenceSec: document.getElementById("inference-section"),
  trainingSec: document.getElementById("training-section"),
  modelStatus: document.getElementById("modelStatus"),

  // Inference mode
  inferenceUpload: document.getElementById("inferenceUpload"),
  inferenceFileInput: document.getElementById("inferenceFileInput"),
  inferencePreview: document.getElementById("inferencePreview"),
  classifyBtn: document.getElementById("classifyBtn"),
  clearInferenceBtn: document.getElementById("clearInferenceBtn"),
  inferenceResults: document.getElementById("inferenceResults"),

  // Training mode
  covidUpload: document.getElementById("covidUpload"),
  covidFileInput: document.getElementById("covidFileInput"),
  covidPreview: document.getElementById("covidPreview"),
  covidCount: document.getElementById("covidCount"),
  normalUpload: document.getElementById("normalUpload"),
  normalFileInput: document.getElementById("normalFileInput"),
  normalPreview: document.getElementById("normalPreview"),
  normalCount: document.getElementById("normalCount"),

  // Training controls
  learningRate: document.getElementById("learningRate"),
  epochs: document.getElementById("epochs"),
  batchSize: document.getElementById("batchSize"),
  validationSplit: document.getElementById("validationSplit"),
  trainBtn: document.getElementById("trainBtn"),
  stopTrainingBtn: document.getElementById("stopTrainingBtn"),

  // Training progress
  trainingProgress: document.getElementById("trainingProgress"),
  progressFill: document.getElementById("progressFill"),
  progressText: document.getElementById("progressText"),
  trainingMetrics: document.getElementById("trainingMetrics"),
  metricLoss: document.getElementById("metricLoss"),
  metricAccuracy: document.getElementById("metricAccuracy"),
  metricValLoss: document.getElementById("metricValLoss"),
  metricValAccuracy: document.getElementById("metricValAccuracy"),

  // Model management
  saveModelBtn: document.getElementById("saveModelBtn"),
  loadModelBtn: document.getElementById("loadModelBtn"),
  exportResultsBtn: document.getElementById("exportResultsBtn"),
}

/**
 * Initialize application
 */
async function init() {
  console.log("Initializing COVID-19 X-Ray Classifier...")
  console.log("TensorFlow.js version:", tf.version.tfjs)

  setupEventListeners()
  updateModelStatus("idle")

  // Try to load existing model
  try {
    await classifier.loadModel()
    appState.modelLoaded = true
    updateModelStatus("ready")
    elements.classifyBtn.disabled = false
  } catch (error) {
    console.log("No saved model found. Ready for training or upload.")
  }
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
  // Mode switching
  elements.modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode))
  })

  // Inference mode
  elements.inferenceUpload.addEventListener("click", () => elements.inferenceFileInput.click())
  elements.inferenceFileInput.addEventListener("change", handleInferenceUpload)
  setupDragAndDrop(elements.inferenceUpload, handleInferenceFiles)
  elements.classifyBtn.addEventListener("click", handleClassify)
  elements.clearInferenceBtn.addEventListener("click", clearInferenceImages)

  // Training mode uploads
  elements.covidUpload.addEventListener("click", () => elements.covidFileInput.click())
  elements.covidFileInput.addEventListener("change", (e) => handleTrainingUpload(e, "covid"))
  setupDragAndDrop(elements.covidUpload, (files) => handleTrainingFiles(files, "covid"))

  elements.normalUpload.addEventListener("click", () => elements.normalFileInput.click())
  elements.normalFileInput.addEventListener("change", (e) => handleTrainingUpload(e, "normal"))
  setupDragAndDrop(elements.normalUpload, (files) => handleTrainingFiles(files, "normal"))

  // Training controls
  elements.trainBtn.addEventListener("click", handleTraining)
  elements.stopTrainingBtn.addEventListener("click", stopTraining)

  // Model management
  elements.saveModelBtn.addEventListener("click", saveModel)
  elements.loadModelBtn.addEventListener("click", loadModel)
  elements.exportResultsBtn.addEventListener("click", exportResults)

  // Update train button state when images change
  ;[elements.covidFileInput, elements.normalFileInput].forEach((input) => {
    input.addEventListener("change", updateTrainButtonState)
  })
}

/**
 * Setup drag and drop for upload areas
 */
function setupDragAndDrop(element, handler) {
  element.addEventListener("dragover", (e) => {
    e.preventDefault()
    element.classList.add("dragover")
  })

  element.addEventListener("dragleave", () => {
    element.classList.remove("dragover")
  })

  element.addEventListener("drop", (e) => {
    e.preventDefault()
    element.classList.remove("dragover")
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"))
    if (files.length > 0) {
      handler(files)
    }
  })
}

/**
 * Switch between inference and training modes
 */
function switchMode(mode) {
  appState.mode = mode

  elements.modeBtns.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode)
  })

  elements.inferenceSec.classList.toggle("active", mode === "inference")
  elements.trainingSec.classList.toggle("active", mode === "training")
}

/**
 * Update model status indicator
 */
function updateModelStatus(status) {
  const statusMap = {
    idle: { text: "Model Not Loaded", class: "status-idle" },
    ready: { text: "Model Ready", class: "status-ready" },
    training: { text: "Training...", class: "status-training" },
  }

  const config = statusMap[status]
  elements.modelStatus.className = `status-indicator ${config.class}`
  elements.modelStatus.innerHTML = `<span>●</span> ${config.text}`
}

/**
 * Handle inference image upload
 */
async function handleInferenceUpload(e) {
  const files = Array.from(e.target.files)
  await handleInferenceFiles(files)
}

/**
 * Process inference image files
 */
async function handleInferenceFiles(files) {
  for (const file of files) {
    try {
      const img = await dataLoader.loadImageFromFile(file)
      const tensor = dataLoader.preprocessImage(img)

      appState.inferenceImages.push({
        file,
        image: img,
        tensor,
        url: URL.createObjectURL(file),
      })

      addImagePreview(img, file.name, "inference")
    } catch (error) {
      console.error("Error loading image:", error)
      alert(`Failed to load ${file.name}: ${error.message}`)
    }
  }

  elements.classifyBtn.disabled = appState.inferenceImages.length === 0 || !appState.modelLoaded
}

/**
 * Add image preview to grid
 */
function addImagePreview(image, name, type) {
  const container =
    type === "inference" ? elements.inferencePreview : type === "covid" ? elements.covidPreview : elements.normalPreview

  const item = document.createElement("div")
  item.className = "image-preview-item"

  const canvas = dataLoader.createPreviewCanvas(image, 200)
  item.appendChild(canvas)

  const removeBtn = document.createElement("button")
  removeBtn.className = "remove-btn"
  removeBtn.innerHTML = "×"
  removeBtn.onclick = () => {
    item.remove()
    removeImageFromState(name, type)
  }
  item.appendChild(removeBtn)

  container.appendChild(item)
}

/**
 * Remove image from state
 */
function removeImageFromState(name, type) {
  if (type === "inference") {
    const index = appState.inferenceImages.findIndex((img) => img.file.name === name)
    if (index !== -1) {
      appState.inferenceImages[index].tensor.dispose()
      URL.revokeObjectURL(appState.inferenceImages[index].url)
      appState.inferenceImages.splice(index, 1)
    }
    elements.classifyBtn.disabled = appState.inferenceImages.length === 0
  } else if (type === "covid") {
    const index = appState.covidImages.findIndex((img) => img.file.name === name)
    if (index !== -1) {
      appState.covidImages[index].tensor.dispose()
      appState.covidImages.splice(index, 1)
    }
    elements.covidCount.textContent = appState.covidImages.length
  } else if (type === "normal") {
    const index = appState.normalImages.findIndex((img) => img.file.name === name)
    if (index !== -1) {
      appState.normalImages[index].tensor.dispose()
      appState.normalImages.splice(index, 1)
    }
    elements.normalCount.textContent = appState.normalImages.length
  }
  updateTrainButtonState()
}

/**
 * Clear all inference images
 */
function clearInferenceImages() {
  appState.inferenceImages.forEach((img) => {
    img.tensor.dispose()
    URL.revokeObjectURL(img.url)
  })
  appState.inferenceImages = []
  elements.inferencePreview.innerHTML = ""
  elements.inferenceResults.innerHTML = ""
  elements.classifyBtn.disabled = true
}

/**
 * Handle classification
 */
async function handleClassify() {
  if (appState.inferenceImages.length === 0 || !appState.modelLoaded) {
    return
  }

  elements.classifyBtn.disabled = true
  elements.classifyBtn.innerHTML = '<span class="spinner"></span> Classifying...'
  elements.inferenceResults.innerHTML = ""

  try {
    // Stack all images into batch
    const imagesBatch = tf.stack(appState.inferenceImages.map((img) => img.tensor))

    // Get predictions
    const predictions = await classifier.predict(imagesBatch)

    // Display results
    displayResults(predictions, appState.inferenceImages)

    // Cleanup
    imagesBatch.dispose()
  } catch (error) {
    console.error("Classification error:", error)
    alert("Error during classification: " + error.message)
  } finally {
    elements.classifyBtn.disabled = false
    elements.classifyBtn.innerHTML = "<span>🔍</span> Classify Images"
  }
}

/**
 * Display classification results
 */
function displayResults(predictions, images) {
  elements.inferenceResults.innerHTML = '<h3 style="margin-bottom: 1rem;">Classification Results</h3>'

  predictions.forEach((pred, idx) => {
    const img = images[idx]
    const isCovid = pred.class === "COVID-19"
    const confidence = pred.confidence

    // Determine confidence level
    let confidenceLevel, confidenceClass
    if (confidence >= 0.8) {
      confidenceLevel = "High"
      confidenceClass = "confidence-high"
    } else if (confidence >= 0.6) {
      confidenceLevel = "Medium"
      confidenceClass = "confidence-medium"
    } else {
      confidenceLevel = "Low"
      confidenceClass = "confidence-low"
    }

    const card = document.createElement("div")
    card.className = `result-card ${isCovid ? "covid" : "normal"}`
    card.innerHTML = `
            <div class="result-header">
                <div>
                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">
                        ${img.file.name}
                    </div>
                    <div class="result-label" style="color: ${isCovid ? "var(--danger-color)" : "var(--success-color)"}">
                        ${pred.class}
                    </div>
                </div>
                <div class="confidence-badge ${confidenceClass}">
                    ${confidenceLevel} Confidence
                </div>
            </div>
            <div class="confidence-bars">
                <div class="confidence-bar-item">
                    <div class="confidence-bar-label">COVID-19</div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width: ${pred.probabilities["COVID-19"] * 100}%; background: var(--danger-color);">
                            ${(pred.probabilities["COVID-19"] * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                <div class="confidence-bar-item">
                    <div class="confidence-bar-label">Normal</div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width: ${pred.probabilities["Normal"] * 100}%; background: var(--success-color);">
                            ${(pred.probabilities["Normal"] * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>
            <div class="warning-box" style="margin-top: 1rem; padding: 1rem;">
                <p style="font-size: 0.875rem; margin: 0;">
                    ⚠️ This is an educational demonstration only. Never use for actual medical diagnosis.
                </p>
            </div>
        `

    elements.inferenceResults.appendChild(card)
  })
}

/**
 * Handle training image upload
 */
async function handleTrainingUpload(e, type) {
  const files = Array.from(e.target.files)
  await handleTrainingFiles(files, type)
}

/**
 * Process training image files
 */
async function handleTrainingFiles(files, type) {
  const targetArray = type === "covid" ? appState.covidImages : appState.normalImages

  for (const file of files) {
    try {
      const img = await dataLoader.loadImageFromFile(file)
      const tensor = dataLoader.preprocessImage(img)

      targetArray.push({
        file,
        image: img,
        tensor,
        url: URL.createObjectURL(file),
      })

      addImagePreview(img, file.name, type)
    } catch (error) {
      console.error("Error loading image:", error)
      alert(`Failed to load ${file.name}: ${error.message}`)
    }
  }

  if (type === "covid") {
    elements.covidCount.textContent = appState.covidImages.length
  } else {
    elements.normalCount.textContent = appState.normalImages.length
  }

  updateTrainButtonState()
}

/**
 * Update train button state
 */
function updateTrainButtonState() {
  const hasEnoughData = appState.covidImages.length >= 5 && appState.normalImages.length >= 5
  elements.trainBtn.disabled = !hasEnoughData || appState.isTraining
}

/**
 * Handle model training
 */
async function handleTraining() {
  if (appState.covidImages.length < 5 || appState.normalImages.length < 5) {
    alert("Please upload at least 5 images for each class (COVID-19 and Normal)")
    return
  }

  appState.isTraining = true
  elements.trainBtn.style.display = "none"
  elements.stopTrainingBtn.style.display = "inline-flex"
  elements.trainingProgress.style.display = "block"
  elements.trainingMetrics.style.display = "grid"
  updateModelStatus("training")

  try {
    // Prepare training data
    elements.progressText.textContent = "Preparing training data..."

    const covidFiles = appState.covidImages.map((img) => img.file)
    const normalFiles = appState.normalImages.map((img) => img.file)

    const trainingData = await dataLoader.prepareTrainingData(covidFiles, normalFiles, (progress, message) => {
      updateProgress(progress * 100, message)
    })

    appState.trainingData = trainingData

    // Create and compile model
    elements.progressText.textContent = "Creating model..."
    classifier.createModel()
    classifier.compileModel(Number.parseFloat(elements.learningRate.value))

    // Train model
    const epochs = Number.parseInt(elements.epochs.value)
    const batchSize = Number.parseInt(elements.batchSize.value)
    const validationSplit = Number.parseFloat(elements.validationSplit.value)

    await classifier.trainModel(trainingData.images, trainingData.labels, {
      epochs,
      batchSize,
      validationSplit,
      onEpochEnd: async (epoch, logs) => {
        const progress = ((epoch + 1) / epochs) * 100
        updateProgress(progress, `Epoch ${epoch + 1}/${epochs}`)
        updateMetrics(logs)

        // Visualize training progress
        if (epoch === 0) {
          tfvis.visor().open()
        }

        const container = { name: "Training Progress", tab: "Training" }
        tfvis.show.history(container, classifier.trainingHistory, ["loss", "valLoss"])

        const accuracyContainer = { name: "Accuracy", tab: "Training" }
        tfvis.show.history(accuracyContainer, classifier.trainingHistory, ["accuracy", "valAccuracy"])
      },
    })

    // Training complete
    appState.modelLoaded = true
    elements.saveModelBtn.disabled = false
    elements.exportResultsBtn.disabled = false
    elements.classifyBtn.disabled = false

    alert("Training completed successfully! You can now use the model for inference or save it.")
  } catch (error) {
    console.error("Training error:", error)
    alert("Error during training: " + error.message)
  } finally {
    appState.isTraining = false
    elements.trainBtn.style.display = "inline-flex"
    elements.stopTrainingBtn.style.display = "none"
    updateModelStatus(appState.modelLoaded ? "ready" : "idle")
    updateTrainButtonState()
  }
}

/**
 * Stop training
 */
function stopTraining() {
  if (classifier.model) {
    classifier.model.stopTraining = true
  }
}

/**
 * Update progress bar
 */
function updateProgress(percent, message) {
  elements.progressFill.style.width = `${percent}%`
  elements.progressFill.textContent = `${Math.round(percent)}%`
  elements.progressText.textContent = message
}

/**
 * Update training metrics display
 */
function updateMetrics(logs) {
  elements.metricLoss.textContent = logs.loss.toFixed(4)
  elements.metricAccuracy.textContent = (logs.acc * 100).toFixed(2) + "%"
  elements.metricValLoss.textContent = logs.val_loss.toFixed(4)
  elements.metricValAccuracy.textContent = (logs.val_acc * 100).toFixed(2) + "%"
}

/**
 * Save model to local storage
 */
async function saveModel() {
  try {
    await classifier.saveModel()
    alert("Model saved successfully to browser storage!")
  } catch (error) {
    console.error("Error saving model:", error)
    alert("Failed to save model: " + error.message)
  }
}

/**
 * Load model from local storage
 */
async function loadModel() {
  try {
    await classifier.loadModel()
    appState.modelLoaded = true
    updateModelStatus("ready")
    elements.classifyBtn.disabled = false
    alert("Model loaded successfully!")
  } catch (error) {
    console.error("Error loading model:", error)
    alert("Failed to load model. Please train a model first.")
  }
}

/**
 * Export training results
 */
function exportResults() {
  if (!classifier.trainingHistory || classifier.trainingHistory.length === 0) {
    alert("No training results to export")
    return
  }

  // Create CSV content
  let csv = "Epoch,Loss,Accuracy,Validation Loss,Validation Accuracy\n"
  classifier.trainingHistory.forEach((record) => {
    csv += `${record.epoch},${record.loss},${record.accuracy},${record.valLoss},${record.valAccuracy}\n`
  })

  // Download CSV
  const blob = new Blob([csv], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = "training-results.csv"
  a.click()
  URL.revokeObjectURL(url)
}

// Initialize app when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init)
} else {
  init()
}

// Cleanup on page unload
window.addEventListener("beforeunload", () => {
  dataLoader.dispose()
  classifier.dispose()
})
