# Classical Deepfake Detection with Handcrafted Image Features

Pattern Recognition and Machine Learning — Assignment 1, Task A.

This project investigates whether classical machine-learning classifiers can distinguish real face images from manipulated face images using handcrafted visual features rather than a deep neural network.

## Project overview

The main experiment uses extracted frames from FaceForensics++ at the C23/C32 compression setting, distributed through the Kaggle dataset [FaceForensics Extracted Dataset C23](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23).

The notebook:

- samples 1,000 real images and 1,000 fake images;
- balances fake images across five manipulation methods:
  Deepfakes, Face2Face, FaceShifter, FaceSwap, and NeuralTextures;
- resizes images to `256 x 256`;
- extracts 222 handcrafted features from colour statistics, colour histograms, entropy, texture and frequency information;
- standardises features with `StandardScaler`;
- creates train, validation and test splits;
- compares k-nearest neighbours, Naive Bayes, Logistic Regression, SVM and Decision Tree classifiers;
- visualises the feature space with PCA, t-SNE and UMAP;
- evaluates the selected model on a small external dataset to examine performance under distribution shift.


## Dataset

### FaceForensics++ sample

The Kaggle download contains 30,000 extracted frames:

| Original | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 |

For computational efficiency, the experiment uses:

| Binary class | Samples |
|---|---:|
| Real | 1,000 |
| Fake | 1,000 |

The fake class contains 200 samples from each manipulation method.

### External dataset

`own_dataset.zip` contains 40 additional images: 20 real and 20 fake images. The fake images were generated with Faceswapper.ai. This dataset is used only for a distribution-shift check and is not mixed with the FaceForensics++ training data.

The datasets are not included in this repository's source code. Download or provide them separately before running the relevant notebook cells. The Kaggle dataset is listed as **CC BY-NC 4.0** on its dataset page; check the original dataset terms before redistribution or commercial use.

## Feature extraction

Each image is resized to `256 x 256` and represented by 222 handcrafted features, including:

- per-channel means and standard deviations;
- 32-bin RGB colour histograms;
- per-channel entropy;
- texture/statistical descriptors;
- FFT-based frequency-domain moments.

The feature matrix is standardised using a `StandardScaler` fitted only on the training split to avoid data leakage.

## Models

The following classifiers are compared:

1. k-nearest neighbours;
2. Gaussian/Bernoulli Naive Bayes;
3. Logistic Regression;
4. Support Vector Machine;
5. Decision Tree;
6. a most-frequent dummy classifier baseline.

Evaluation uses accuracy, macro F1-score, macro precision, macro recall and confusion matrices.

## Results

The validation set selected k-NN as the best model. On the held-out test set, the Decision Tree achieved the highest accuracy among the tested models, although performance for all models was close to random classification.

| Model | Test accuracy | Test macro F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|
| k-NN | 0.5333 | 0.5316 | 0.5338 | 0.5333 |
| Naive Bayes | 0.5067 | 0.5066 | 0.5067 | 0.5067 |
| Logistic Regression | 0.4733 | 0.4725 | 0.4732 | 0.4733 |
| SVM | 0.5333 | 0.5326 | 0.5335 | 0.5333 |
| Decision Tree | **0.5433** | **0.5432** | **0.5434** | **0.5433** |

The notebook reports 34 test samples misclassified by all five classifiers. These results suggest that the handcrafted low-level features do not separate real and manipulated faces reliably enough for this dataset and task.

## Reproducibility

### Requirements

Use Python 3.10 or newer and install the notebook dependencies:

```bash
pip install numpy pandas scipy opencv-python pillow matplotlib seaborn scikit-learn umap-learn kaggle
```

### Run in Google Colab

1. Open `deepfake_detection.ipynb` in Google Colab.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets, or configure the Kaggle API manually.
3. Run the notebook from top to bottom.
4. Upload `own_dataset.zip` when the external-dataset section requests it.

The notebook downloads and extracts the large Kaggle dataset automatically. The download is approximately 3.8 GB, so the complete experiment requires sufficient disk space and runtime.

## Limitations

- The experiment uses a computationally limited sample instead of all available frames.
- The handcrafted features capture low-level image statistics and may miss semantic or spatial manipulation artifacts.
- Frames from the same source videos can be correlated.
- The external dataset contains only 40 images and was produced with one manipulation tool.
- The measured results should not be interpreted as a production-ready deepfake detector.

## Citation and attribution

- [FaceForensics Extracted Dataset C23 on Kaggle](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23).

