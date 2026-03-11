# Gesture Recognition Application Using Feature Extraction

## Submission
This project implements a real-time gesture recognition system using motion sensor data from an Arduino Nano 33 BLE Sense Rev2.  

## Hardware
- Arduino Nano 33 BLE Sense Rev2
- On-board IMU (accelerometer + gyroscope)

## Project Overview
1. Data collection from the IMU sensor
2. Feature extraction in Google Colab
3. Model training using a neural network
4. Conversion of the trained model to TensorFlow Lite
5. Deployment on Arduino for on-device inference

## Gesture Classes
The dataset was collected for the following classes:
- circle
- left_right
- rest
- up_down

During on-device testing, the most stable and reliable classes were:
- circle
- rest
- up_down

The `left_right` class showed occasional confusion during embedded inference.

## Data Collection
Sensor data was collected from the accelerometer and gyroscope using the Arduino Nano 33 BLE Sense Rev2.

Each recording used:
- Window size: 128 samples
- Sensor channels:
  - aX
  - aY
  - aZ
  - gX
  - gY
  - gZ

The collected data was stored in CSV files:
- `Circle.csv`
- `Left-right.csv`
- `Rest.csv`
- `Up-down.csv`

## Feature Extraction
Feature extraction was applied on all 6 IMU axes.

### Time-domain features
For each axis:
- Mean
- Standard deviation
- RMS
- Minimum
- Maximum

### Frequency-domain features
For each axis:
- Dominant frequency
- Peak spectral power

This produced:
- 7 features per axis
- 6 axes
- Total feature vector size = 42 features

In Google Colab, frequency-domain features were extracted using PSD-based analysis.  
For on-device inference, a lightweight approximation of frequency features was implemented to reduce computational cost on the microcontroller.

## Model
A Multi-Layer Perceptron (MLP) classifier was trained using the extracted features.

Architecture:
- Input layer: 42 features
- Dense layer: 96 neurons, ReLU
- Dense layer: 48 neurons, ReLU
- Output layer: 4 neurons, Softmax

## Training Pipeline
The model was trained in Google Colab using:
- Train/Validation/Test split
- StandardScaler normalization
- TensorFlow / Keras

## Results
The trained model achieved excellent classification performance on the test set.

### Test Results
- Test accuracy: **1.00**

### Classification performance
The confusion matrix and classification report showed perfect classification on the test set.

## On-Device Inference
The trained model was converted to TensorFlow Lite and then exported to `model.h` for deployment on Arduino.

The Arduino sketch:
- Waits for motion trigger
- Captures a 128-sample IMU window
- Extracts 42 features
- Normalizes the features
- Runs on-device inference
- Prints the predicted gesture in the Serial Monitor

### Stable on-device outputs observed
- `Prediction: rest`
- `Prediction: up_down`
- `Prediction: circle`

## Limitation
Although the offline test results were perfect, the `left_right` class showed instability during real-time on-device inference.

A possible reason is that the embedded implementation used a lightweight approximation of frequency-domain features instead of the full PSD-based pipeline used in Colab.  
This can affect the separability of similar motion classes such as `left_right` and `circle` or `rest`.

## Files Included
### Data
- `Circle.csv`
- `Left-right.csv`
- `Rest.csv`
- `Up-down.csv`
- 
### Arduino
- Data capture sketch
- Final gesture classifier sketch
- `model.h`

### Results
- `classification_report.txt`
- `confusion_matrix.png`
- `training_curves.png`
- `scaler_values.txt`
