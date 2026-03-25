import streamlit as st
import numpy as np
import tensorflow as tf
from preprocess import extract_features
import os
import pandas as pd
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

MODEL_PATH = "emotion_model.h5"

@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        mean = np.load('scaler_mean.npy')
        scale = np.load('scaler_scale.npy')
        classes = np.load('classes.npy', allow_pickle=True)
        return model, mean, scale, classes
    except Exception as e:
        return None, None, None, None

model, scaler_mean, scaler_scale, classes = load_assets()

st.title("🎙️ Speech Emotion Recognition")
st.write("Detect the emotion of a speaker using our trained 1D CNN model. You can either upload a file or record audio directly from your microphone.")

if model is None:
    st.error("Model or scaler parameters not found. Ensure the dataset has been preprocessed and the model is trained.")
    st.stop()

def predict_audio(file_path):
    # Extract features
    features = extract_features(file_path)
    
    # Standardize
    features_scaled = (features - scaler_mean) / scaler_scale
    
    # Predict
    X_input = np.expand_dims(features_scaled, axis=0)
    predictions = model.predict(X_input)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_emotion = classes[predicted_class_idx].upper()
    return predicted_emotion, predictions

tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav"], key="uploader")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Predict Emotion", key="predict_upload"):
            with st.spinner("Analyzing Audio..."):
                temp_path = "temp_uploaded.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                try:
                    predicted_emotion, predictions = predict_audio(temp_path)
                    st.success(f"**Predicted Emotion:** {predicted_emotion}")
                    
                    # Display bar chart of confidence
                    st.write("### Prediction Confidence")
                    conf_data = pd.DataFrame({
                        "Emotion": [c.capitalize() for c in classes],
                        "Confidence": predictions * 100
                    })
                    st.bar_chart(conf_data.set_index("Emotion"))
                except Exception as e:
                    st.error(f"Error evaluating audio: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

with tab2:
    st.write("Click the microphone icon to start/stop recording.")
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("Predict Emotion (Recording)", key="predict_record"):
            with st.spinner("Analyzing Recording..."):
                temp_path = "temp_record.wav"
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)
                
                try:
                    predicted_emotion, predictions = predict_audio(temp_path)
                    st.success(f"**Predicted Emotion:** {predicted_emotion}")
                    
                    # Display bar chart of confidence
                    st.write("### Prediction Confidence")
                    conf_data = pd.DataFrame({
                        "Emotion": [c.capitalize() for c in classes],
                        "Confidence": predictions * 100
                    })
                    st.bar_chart(conf_data.set_index("Emotion"))
                except Exception as e:
                    st.error(f"Error evaluating audio: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
