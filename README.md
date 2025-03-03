# 3D CNN-Based CCTV Anomaly Detection Model

## Overview
This project implements a 3D Convolutional Neural Network (3D CNN) for CCTV anomaly detection. The model processes video frames as a sequence of images and classifies them into 14 different anomaly categories.

## Model Architecture
The model consists of the following layers:
- **Input Layer**: Accepts input of shape `(64, 224, 224, 3)`, where:
  - `64` represents the number of frames in a video sequence.
  - `224x224` is the spatial resolution of each frame.
  - `3` represents the RGB color channels.
- **3D Convolutional Layers**: Extracts spatial and temporal features from the video input.
- **Batch Normalization Layers**: Normalizes activations to stabilize training.
- **MaxPooling3D Layers**: Reduces spatial and temporal dimensions while retaining essential features.
- **Global Average Pooling**: Aggregates feature maps into a lower-dimensional representation.
- **Dropout Layer**: Helps prevent overfitting.
- **Dense Layer**: Outputs a classification vector of size `14` (number of anomaly classes).

## Model Summary
| Layer Type                  | Output Shape            | Parameters |
|-----------------------------|-------------------------|------------|
| Input Layer                 | (None, 64, 224, 224, 3) | 0          |
| Conv3D                      | (None, 64, 224, 224, 64) | 28,288    |
| BatchNormalization          | (None, 64, 224, 224, 64) | 256       |
| MaxPooling3D                | (None, 64, 112, 112, 64) | 0         |
| Conv3D                      | (None, 64, 112, 112, 128) | 614,528  |
| BatchNormalization          | (None, 64, 112, 112, 128) | 512       |
| MaxPooling3D                | (None, 32, 56, 56, 128) | 0         |
| Conv3D                      | (None, 32, 56, 56, 256) | 884,992   |
| BatchNormalization          | (None, 32, 56, 56, 256) | 1,024     |
| MaxPooling3D                | (None, 16, 28, 28, 256) | 0         |
| Conv3D                      | (None, 16, 28, 28, 512) | 3,539,456 |
| BatchNormalization          | (None, 16, 28, 28, 512) | 2,048     |
| MaxPooling3D                | (None, 8, 14, 14, 512)  | 0         |
| GlobalAveragePooling3D      | (None, 512)             | 0         |
| Dropout                     | (None, 512)             | 0         |
| Dense (Output Layer)        | (None, 14)              | 7,182     |

- **Total Parameters**: 15,231,020 (58.10 MB)
- **Trainable Parameters**: 5,076,366 (19.36 MB)
- **Non-trainable Parameters**: 1,920 (7.50 KB)
- **Optimizer Parameters**: 10,152,734 (38.73 MB)

## Training
- The model is trained using **Adam optimizer**.
- Loss function: **Categorical Crossentropy** (for multi-class classification).
- Metrics: **Accuracy** and **Precision/Recall**.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy opencv-python matplotlib
```

## Usage
1. **Prepare Dataset**: Ensure the dataset is formatted as `(64, 224, 224, 3)` sequences.
2. **Train Model**:
   ```python
   model.fit(train_data, epochs=50, validation_data=val_data)
   ```
3. **Evaluate Model**:
   ```python
   model.evaluate(test_data)
   ```
4. **Make Predictions**:
   ```python
   predictions = model.predict(sample_video)
   ```

## Future Work
- Improve model accuracy using attention mechanisms.
- Optimize for real-time deployment.
- Implement anomaly detection with unsupervised learning methods.

## Contributors
- **Eshwar B.**

## License
This project is open-source under the MIT License.

