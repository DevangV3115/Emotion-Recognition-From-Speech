# Emotion Recognition from Speech

This project implements an Emotion Recognition system from speech audio using the **RAVDESS** dataset. It extracts **MFCC, Chroma, and Mel Spectrogram** features and trains a **1D Convolutional Neural Network (CNN)** for predicting emotions mapping to (neutral, calm, happy, sad, angry, fearful, disgust, surprised).

## Overview of Scripts

1. `download_data.py`: A utility to download and extract the RAVDESS dataset. (Note: Due to Zenodo security, it may be better to download using curl/Invoke-WebRequest if this script fails).
2. `preprocess.py`: Processes the downloaded audio files in the `data/` directory. Extracts acoustic features and labels them based on the filename standard. Saves to `X.npy` and `y.npy`.
3. `train_model.py`: Loads the extracted features, scales them, trains a 1D CNN model, and evaluates its accuracy. Saves the trained model to `emotion_model.h5` and stores standard scaler properties for inference.
4. `predict.py`: Runs emotion prediction on a single, unseen audio `.wav` file.

## How to Run

### Step 1: Install Dependencies
Ensure you have the following installed:
```bash
pip install librosa scikit-learn tensorflow pandas numpy matplotlib seaborn soundfile
```

### Step 2: Prepare the Dataset
If you haven't already extracted the zip file, place `Audio_Speech_Actors_01-24.zip` in the same directory and extract it into a folder named `data`. Ensure that the `.wav` files are accessible inside `data/`.

### Step 3: Extract Features
Run the preprocessing script to generate feature numpy arrays:
```bash
python preprocess.py
```
This will take a few minutes as it processes ~1440 audio files.

### Step 4: Train the Model
Train the CNN model:
```bash
python train_model.py
```
This script will produce `emotion_model.h5`, `classes.npy`, `scaler_mean.npy`, and `scaler_scale.npy`. It also creates a `training_history.png` plot.

### Step 5: Inference
Predict the emotion of a new wav file:
```bash
python predict.py "path_to_some_audio_file.wav"
```
# Emotion-Recognition-From-Speech
