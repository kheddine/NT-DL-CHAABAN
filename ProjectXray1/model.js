/**
 * COVID-19 X-Ray Classification Model
 * Implements CNN architecture for binary classification
 * All operations use TensorFlow.js for browser-based execution
 */

import * as tf from "@tensorflow/tfjs"

class XRayClassifier {
  constructor() {
    this.model = null
    this.isTraining = false
    this.trainingHistory = []
    this.IMAGE_SIZE = 224
    this.NUM_CLASSES = 2
    this.CLASS_NAMES = ["COVID-19", "Normal"]
  }

  /**
   * Creates CNN model architecture
   * Architecture: Conv2D layers with MaxPooling, followed by Dense layers
   * Suitable for medical image binary classification
   */
  createModel() {
    const model = tf.sequential()

    // Input layer: 224x224x3 (RGB images)
    // First convolutional block
    model.add(
      tf.layers.conv2d({
        inputShape: [this.IMAGE_SIZE, this.IMAGE_SIZE, 3],
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal",
      }),
    )
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.batchNormalization())

    // Second convolutional block
    model.add(
      tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal",
      }),
    )
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.batchNormalization())

    // Third convolutional block
    model.add(
      tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal",
      }),
    )
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.batchNormalization())

    // Flatten and dense layers
    model.add(tf.layers.flatten())
    model.add(
      tf.layers.dense({
        units: 128,
        activation: "relu",
        kernelInitializer: "heNormal",
      }),
    )
    model.add(tf.layers.dropout({ rate: 0.5 }))

    // Output layer: 2 classes (COVID-19, Normal)
    model.add(
      tf.layers.dense({
        units: this.NUM_CLASSES,
        activation: "softmax",
      }),
    )

    this.model = model
    console.log("Model created successfully")
    model.summary()
    return model
  }

  /**
   * Compiles model with optimizer and loss function
   * @param {number} learningRate - Learning rate for Adam optimizer
   */
  compileModel(learningRate = 0.001) {
    if (!this.model) {
      throw new Error("Model not created. Call createModel() first.")
    }

    this.model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    })

    console.log("Model compiled with learning rate:", learningRate)
  }

  /**
   * Trains the model on provided dataset
   * @param {tf.Tensor} trainImages - Training images tensor
   * @param {tf.Tensor} trainLabels - Training labels tensor
   * @param {Object} config - Training configuration
   * @returns {Promise} Training history
   */
  async trainModel(trainImages, trainLabels, config = {}) {
    if (!this.model) {
      throw new Error("Model not created or compiled")
    }

    const { epochs = 10, batchSize = 8, validationSplit = 0.2, onEpochEnd = null, onBatchEnd = null } = config

    this.isTraining = true
    this.trainingHistory = []

    try {
      // Configure callbacks
      const callbacks = {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}/${epochs}`)
          console.log(`Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`)
          console.log(`Val Loss: ${logs.val_loss.toFixed(4)}, Val Accuracy: ${logs.val_acc.toFixed(4)}`)

          this.trainingHistory.push({
            epoch: epoch + 1,
            loss: logs.loss,
            accuracy: logs.acc,
            valLoss: logs.val_loss,
            valAccuracy: logs.val_acc,
          })

          if (onEpochEnd) {
            await onEpochEnd(epoch, logs)
          }

          // Yield to UI thread
          await tf.nextFrame()
        },
        onBatchEnd: async (batch, logs) => {
          if (onBatchEnd) {
            await onBatchEnd(batch, logs)
          }
        },
      }

      // Train the model
      const history = await this.model.fit(trainImages, trainLabels, {
        epochs,
        batchSize,
        validationSplit,
        shuffle: true,
        callbacks,
      })

      this.isTraining = false
      console.log("Training completed successfully")

      return history
    } catch (error) {
      this.isTraining = false
      console.error("Training error:", error)
      throw error
    }
  }

  /**
   * Makes predictions on input images
   * @param {tf.Tensor} images - Preprocessed images tensor
   * @returns {Promise<Array>} Predictions with confidence scores
   */
  async predict(images) {
    if (!this.model) {
      throw new Error("Model not loaded. Train or load a model first.")
    }

    return tf.tidy(() => {
      const predictions = this.model.predict(images)
      const probabilities = predictions.arraySync()

      return probabilities.map((probs) => {
        const maxIndex = probs.indexOf(Math.max(...probs))
        return {
          class: this.CLASS_NAMES[maxIndex],
          classIndex: maxIndex,
          confidence: probs[maxIndex],
          probabilities: {
            "COVID-19": probs[0],
            Normal: probs[1],
          },
        }
      })
    })
  }

  /**
   * Evaluates model performance on test data
   * @param {tf.Tensor} testImages - Test images
   * @param {tf.Tensor} testLabels - Test labels
   * @returns {Promise<Object>} Evaluation metrics
   */
  async evaluate(testImages, testLabels) {
    if (!this.model) {
      throw new Error("Model not loaded")
    }

    const result = await this.model.evaluate(testImages, testLabels)
    const loss = await result[0].data()
    const accuracy = await result[1].data()

    return {
      loss: loss[0],
      accuracy: accuracy[0],
    }
  }

  /**
   * Saves model to browser's local storage
   * @param {string} name - Model name
   */
  async saveModel(name = "xray-classifier") {
    if (!this.model) {
      throw new Error("No model to save")
    }

    try {
      await this.model.save(`localstorage://${name}`)
      console.log("Model saved successfully to local storage")

      // Save training history
      localStorage.setItem(`${name}-history`, JSON.stringify(this.trainingHistory))

      return true
    } catch (error) {
      console.error("Error saving model:", error)
      throw error
    }
  }

  /**
   * Loads model from browser's local storage
   * @param {string} name - Model name
   */
  async loadModel(name = "xray-classifier") {
    try {
      this.model = await tf.loadLayersModel(`localstorage://${name}`)
      console.log("Model loaded successfully from local storage")

      // Load training history if available
      const historyJson = localStorage.getItem(`${name}-history`)
      if (historyJson) {
        this.trainingHistory = JSON.parse(historyJson)
      }

      return true
    } catch (error) {
      console.error("Error loading model:", error)
      throw error
    }
  }

  /**
   * Exports model for download
   */
  async downloadModel() {
    if (!this.model) {
      throw new Error("No model to download")
    }

    await this.model.save("downloads://xray-classifier")
    console.log("Model download initiated")
  }

  /**
   * Calculates confusion matrix from predictions
   * @param {Array} predictions - Model predictions
   * @param {Array} trueLabels - True labels
   * @returns {Object} Confusion matrix and metrics
   */
  calculateMetrics(predictions, trueLabels) {
    let tp = 0,
      tn = 0,
      fp = 0,
      fn = 0

    predictions.forEach((pred, idx) => {
      const predicted = pred.classIndex
      const actual = trueLabels[idx]

      if (predicted === 0 && actual === 0) tp++
      else if (predicted === 1 && actual === 1) tn++
      else if (predicted === 0 && actual === 1) fp++
      else if (predicted === 1 && actual === 0) fn++
    })

    const precision = tp / (tp + fp) || 0
    const recall = tp / (tp + fn) || 0
    const f1Score = (2 * (precision * recall)) / (precision + recall) || 0
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0

    return {
      confusionMatrix: { tp, tn, fp, fn },
      precision,
      recall,
      f1Score,
      accuracy,
    }
  }

  /**
   * Disposes of model and frees memory
   */
  dispose() {
    if (this.model) {
      this.model.dispose()
      this.model = null
      console.log("Model disposed, memory freed")
    }
  }

  /**
   * Gets model summary information
   */
  getModelInfo() {
    if (!this.model) {
      return null
    }

    return {
      layers: this.model.layers.length,
      trainable: this.model.trainable,
      parameters: this.model.countParams(),
      inputShape: this.model.inputs[0].shape,
      outputShape: this.model.outputs[0].shape,
    }
  }
}

export { XRayClassifier }
