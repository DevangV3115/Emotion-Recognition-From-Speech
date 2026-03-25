import os
import librosa
import numpy as np
import soundfile as sf
import fnmatch

DATA_DIR = "data"
X_SAVE_PATH = "X.npy"
Y_SAVE_PATH = "y.npy"

# Emotion mapping from RAVDESS filename standard
# (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
EMOTION_MAPPING = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_name):
    """
    Extract MFCC, Chroma, and Mel features from an audio file.
    """
    with sf.SoundFile(file_name) as current_file:
        audio = current_file.read(dtype='float32')
        sample_rate = current_file.samplerate
        
        # In case of stereo audio, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        result = np.array([])
        
        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        
        # Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        
    return result

def load_data():
    X, y = [], []
    print("Loading data...")
    count = 0
    
    # Iterate through all wav files in DATA_DIR
    for root, dirs, files in os.walk(DATA_DIR):
        for file in fnmatch.filter(files, '*.wav'):
            file_path = os.path.join(root, file)
            
            # The RAVDESS file format is like: 03-01-06-01-02-01-12.wav
            parts = file.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                
                # We can also choose to filter certain emotions if needed, 
                # but we'll keep all for this task.
                if emotion_code in EMOTION_MAPPING:
                    emotion = EMOTION_MAPPING[emotion_code]
                    
                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        y.append(emotion)
                        count += 1
                        if count % 100 == 0:
                            print(f"Processed {count} files...")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        
    print(f"Total files processed: {count}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found. Please download and extract dataset first.")
    else:
        X, y = load_data()
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Save extracted features
        np.save(X_SAVE_PATH, X)
        np.save(Y_SAVE_PATH, y)
        print(f"Features and labels saved to {X_SAVE_PATH} and {Y_SAVE_PATH}")
