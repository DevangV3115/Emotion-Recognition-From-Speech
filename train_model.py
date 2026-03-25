import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

X_PATH = "X.npy"
Y_PATH = "y.npy"
MODEL_SAVE_PATH = "emotion_model.h5"

def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        # Add a channel dimension for CNN1D
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def plot_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

def main():
    try:
        X = np.load(X_PATH)
        y = np.load(Y_PATH)
    except FileNotFoundError:
        print(f"Could not load {X_PATH} or {Y_PATH}. Please run preprocess.py first.")
        return

    print(f"Loaded {X.shape[0]} samples.")
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    
    # Save the classes so we can decode them during inference
    np.save('classes.npy', encoder.classes_)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler mean and scale for inference
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    
    # Build model
    input_shape = X_train.shape[1]
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Train
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=100, 
                        batch_size=32, 
                        callbacks=[early_stop])
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    plot_history(history)

if __name__ == "__main__":
    main()
