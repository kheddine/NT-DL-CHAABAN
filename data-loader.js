/**
 * Data Loading and Preprocessing Module
 * Handles image loading, preprocessing, and augmentation for X-ray images
 * All operations maintain medical image quality standards
 */

import * as tf from "@tensorflow/tfjs"

class DataLoader {
  constructor(imageSize = 224) {
    this.imageSize = imageSize
    this.loadedImages = []
  }

  /**
   * Loads image from file and returns as HTMLImageElement
   * @param {File} file - Image file
   * @returns {Promise<HTMLImageElement>}
   */
  async loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      if (!file.type.startsWith("image/")) {
        reject(new Error("File is not an image"))
        return
      }

      // Check file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        reject(new Error("Image file too large (max 10MB)"))
        return
      }

      const reader = new FileReader()
      const img = new Image()

      reader.onload = (e) => {
        img.onload = () => resolve(img)
        img.onerror = () => reject(new Error("Failed to load image"))
        img.crossOrigin = "anonymous" // Prevent CORS issues
        img.src = e.target.result
      }

      reader.onerror = () => reject(new Error("Failed to read file"))
      reader.readAsDataURL(file)
    })
  }

  /**
   * Preprocesses image for model input
   * Resizes to target size, normalizes pixel values to [0,1]
   * Handles both grayscale and RGB images
   * @param {HTMLImageElement} image - Input image
   * @returns {tf.Tensor3D} Preprocessed image tensor
   */
  preprocessImage(image) {
    return tf.tidy(() => {
      // Convert image to tensor
      let tensor = tf.browser.fromPixels(image)

      // Resize to model input size
      tensor = tf.image.resizeBilinear(tensor, [this.imageSize, this.imageSize])

      // Normalize pixel values to [0, 1]
      tensor = tensor.div(255.0)

      // Ensure 3 channels (RGB)
      if (tensor.shape[2] === 1) {
        // Grayscale to RGB by repeating channel
        tensor = tf.tile(tensor, [1, 1, 3])
      } else if (tensor.shape[2] === 4) {
        // RGBA to RGB by dropping alpha channel
        tensor = tensor.slice([0, 0, 0], [this.imageSize, this.imageSize, 3])
      }

      return tensor
    })
  }

  /**
   * Loads and preprocesses multiple images
   * @param {FileList|Array<File>} files - Image files
   * @param {Function} progressCallback - Progress callback
   * @returns {Promise<Object>} Processed images and metadata
   */
  async loadImages(files, progressCallback = null) {
    const images = []
    const metadata = []
    const total = files.length

    for (let i = 0; i < total; i++) {
      try {
        const file = files[i]
        const img = await this.loadImageFromFile(file)
        const tensor = this.preprocessImage(img)

        images.push(tensor)
        metadata.push({
          name: file.name,
          size: file.size,
          type: file.type,
          dimensions: { width: img.width, height: img.height },
          url: URL.createObjectURL(file),
        })

        if (progressCallback) {
          progressCallback((i + 1) / total, file.name)
        }
      } catch (error) {
        console.error(`Error loading ${files[i].name}:`, error)
      }
    }

    return { images, metadata }
  }

  /**
   * Creates batched dataset from images and labels
   * @param {Array<tf.Tensor>} images - Image tensors
   * @param {Array<number>} labels - Class labels (0 or 1)
   * @returns {Object} Batched tensors
   */
  createDataset(images, labels) {
    return tf.tidy(() => {
      // Stack images into single tensor
      const imagesTensor = tf.stack(images)

      // Convert labels to one-hot encoding
      const labelsTensor = tf.oneHot(tf.tensor1d(labels, "int32"), 2)

      return {
        images: imagesTensor,
        labels: labelsTensor,
      }
    })
  }

  /**
   * Applies data augmentation to images
   * Includes random rotation, flipping, and brightness adjustment
   * @param {tf.Tensor4D} images - Batch of images
   * @returns {tf.Tensor4D} Augmented images
   */
  augmentImages(images) {
    return tf.tidy(() => {
      let augmented = images

      // Random horizontal flip (50% chance)
      if (Math.random() > 0.5) {
        augmented = tf.image.flipLeftRight(augmented)
      }

      // Random brightness adjustment (-0.2 to +0.2)
      const brightnessDelta = (Math.random() - 0.5) * 0.4
      augmented = tf.image.adjustBrightness(augmented, brightnessDelta)

      // Clip values to [0, 1] range
      augmented = tf.clipByValue(augmented, 0, 1)

      return augmented
    })
  }

  /**
   * Splits dataset into training and validation sets
   * @param {tf.Tensor} images - All images
   * @param {tf.Tensor} labels - All labels
   * @param {number} validationSplit - Fraction for validation (0-1)
   * @returns {Object} Split datasets
   */
  splitDataset(images, labels, validationSplit = 0.2) {
    return tf.tidy(() => {
      const numExamples = images.shape[0]
      const numValidation = Math.floor(numExamples * validationSplit)
      const numTrain = numExamples - numValidation

      // Shuffle indices
      const indices = tf.util.createShuffledIndices(numExamples)

      // Split images
      const trainIndices = indices.slice(0, numTrain)
      const valIndices = indices.slice(numTrain)

      const trainImages = tf.gather(images, trainIndices)
      const trainLabels = tf.gather(labels, trainIndices)
      const valImages = tf.gather(images, valIndices)
      const valLabels = tf.gather(labels, valIndices)

      return {
        train: { images: trainImages, labels: trainLabels },
        validation: { images: valImages, labels: valLabels },
      }
    })
  }

  /**
   * Prepares training data from COVID and Normal image sets
   * @param {Array<File>} covidFiles - COVID-19 positive X-rays
   * @param {Array<File>} normalFiles - Normal X-rays
   * @param {Function} progressCallback - Progress callback
   * @returns {Promise<Object>} Prepared dataset
   */
  async prepareTrainingData(covidFiles, normalFiles, progressCallback = null) {
    console.log(`Preparing training data: ${covidFiles.length} COVID, ${normalFiles.length} Normal`)

    // Load COVID images (label 0)
    const covidData = await this.loadImages(covidFiles, (progress, name) => {
      if (progressCallback) {
        progressCallback(progress * 0.5, `Loading COVID images: ${name}`)
      }
    })

    // Load Normal images (label 1)
    const normalData = await this.loadImages(normalFiles, (progress, name) => {
      if (progressCallback) {
        progressCallback(0.5 + progress * 0.5, `Loading Normal images: ${name}`)
      }
    })

    // Create labels
    const covidLabels = new Array(covidData.images.length).fill(0)
    const normalLabels = new Array(normalData.images.length).fill(1)

    // Combine datasets
    const allImages = [...covidData.images, ...normalData.images]
    const allLabels = [...covidLabels, ...normalLabels]
    const allMetadata = [...covidData.metadata, ...normalData.metadata]

    // Create tensors
    const dataset = this.createDataset(allImages, allLabels)

    // Clean up individual tensors
    covidData.images.forEach((t) => t.dispose())
    normalData.images.forEach((t) => t.dispose())

    return {
      images: dataset.images,
      labels: dataset.labels,
      metadata: allMetadata,
      counts: {
        covid: covidFiles.length,
        normal: normalFiles.length,
        total: allImages.length,
      },
    }
  }

  /**
   * Creates preview canvas for image
   * @param {HTMLImageElement} image - Source image
   * @param {number} size - Canvas size
   * @returns {HTMLCanvasElement}
   */
  createPreviewCanvas(image, size = 200) {
    const canvas = document.createElement("canvas")
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext("2d")

    // Calculate scaling to fit image in square
    const scale = Math.min(size / image.width, size / image.height)
    const scaledWidth = image.width * scale
    const scaledHeight = image.height * scale
    const x = (size - scaledWidth) / 2
    const y = (size - scaledHeight) / 2

    // Draw with black background
    ctx.fillStyle = "#000"
    ctx.fillRect(0, 0, size, size)
    ctx.drawImage(image, x, y, scaledWidth, scaledHeight)

    return canvas
  }

  /**
   * Cleans up resources
   */
  dispose() {
    this.loadedImages.forEach((img) => {
      if (img.url) {
        URL.revokeObjectURL(img.url)
      }
    })
    this.loadedImages = []
  }
}

export { DataLoader }
