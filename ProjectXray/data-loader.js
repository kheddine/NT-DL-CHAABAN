/**
 * data-loader.js
 * Handles image upload, preprocessing, normalization, and augmentation.
 * All operations happen client-side for privacy.
 */

let uploadedImages = [];

/** Load and preprocess an image file to tensor [224,224,3] normalized [0,1] */
async function preprocessImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const tensor = tf.tidy(() => {
          const t = tf.browser.fromPixels(img);
          const resized = tf.image.resizeBilinear(t, [224, 224]);
          const rgb = resized.shape[2] === 1 ? resized.expandDims(2).concat(resized, 2).concat(resized, 2) : resized;
          const normalized = rgb.toFloat().div(tf.scalar(255));
          return normalized;
        });
        resolve(tensor);
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** Basic augmentation: flip, rotate, brightness */
function augmentImage(tensor) {
  return tf.tidy(() => {
    let t = tensor;
    if (Math.random() > 0.5) t = tf.image.flipLeftRight(t);
    if (Math.random() > 0.5) t = tf.image.flipUpDown(t);
    const angle = (Math.random() - 0.5) * 0.2; // ±0.2 radians
    t = tf.image.rotateWithOffset(t, angle, 0);
    const brightness = (Math.random() - 0.5) * 0.2;
    t = tf.image.adjustBrightness(t, brightness);
    return t;
  });
}

/** Create TensorFlow.js dataset (X, Y) */
async function createDataset(files, labels) {
  const tensors = [];
  const oneHots = [];

  for (let i = 0; i < files.length; i++) {
    const imgTensor = await preprocessImage(files[i]);
    const augTensor = augmentImage(imgTensor);
    const y = tf.oneHot(tf.tensor1d([labels[i]], 'int32'), 2);
    tensors.push(augTensor);
    oneHots.push(y);
    tf.dispose(imgTensor);
  }

  const xs = tf.stack(tensors);
  const ys = tf.concat(oneHots);
  tensors.forEach(t => t.dispose());
  oneHots.forEach(y => y.dispose());
  return { xs, ys };
}

/** Utility: Clear memory of uploaded images */
function clearUploadedImages() {
  uploadedImages = [];
  tf.engine().memory().numTensors && tf.disposeVariables();
}
