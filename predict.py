import sys
import os
import numpy as np
import tensorflow as tf
from preprocess import extract_features

MODEL_PATH = "emotion_model.h5"

def load_scaler():
    try:
        mean = np.load('scaler_mean.npy')
        scale = np.load('scaler_scale.npy')
        return mean, scale
    except FileNotFoundError:
        print("Scaler parameters not found. Ensure the model has been trained.")
        sys.exit(1)

def load_classes():
    try:
        classes = np.load('classes.npy', allow_pickle=True)
        return classes
    except FileNotFoundError:
        print("Classes file not found. Ensure the model has been trained.")
        sys.exit(1)

def main(wav_file_path):
    if not os.path.exists(wav_file_path):
        print(f"File not found: {wav_file_path}")
        return
        
    print(f"Processing audio file: {wav_file_path}")
    
    # Extract features
    try:
        features = extract_features(wav_file_path)
    except Exception as e:
        print(f"Failed to extract features: {e}")
        return
        
    # Standardize
    mean, scale = load_scaler()
    features_scaled = (features - mean) / scale
    
    # Reshape for CNN
    # Shape: (1, num_features)
    X_input = np.expand_dims(features_scaled, axis=0)
    
    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        return
        
    # Predict
    predictions = model.predict(X_input)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Convert index to string label
    classes = load_classes()
    predicted_emotion = classes[predicted_class_idx]
    
    print("\n--- Prediction Results ---")
    for i, c in enumerate(classes):
        print(f"{c.capitalize()}: {predictions[0][i]*100:.2f}%")
        
    print(f"\nPredicted Emotion >>> {predicted_emotion.upper()} <<<")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_wav_file>")
    else:
        main(sys.argv[1])
