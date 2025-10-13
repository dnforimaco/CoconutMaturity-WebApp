import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import os

# --- CONFIGURATION (MUST MATCH TRAINING CONFIG) ---
SAMPLING_RATE = 44100
N_MFCC = 40
MAX_MFCC_LENGTH = 100 
NUM_CLASSES = 3
TARGET_NAMES = {0: 'Premature', 1: 'Mature', 2: 'Overmature'}
MODEL_PATH = 'attention_lstm_model.h5'

# Replace with the actual names of your three WAV files
WAV_FILES = [
    'coconut_tap_A.wav', 
    'coconut_tap_B.wav', 
    'coconut_tap_C.wav'
]

# --- 1. Custom Attention Layer (Needed to load the custom Keras model) ---
class Attention(Layer):
    """Custom Keras Global Attention Layer."""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True, name='W')
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = K.dot(inputs, self.W)
        alpha = K.softmax(K.squeeze(score, axis=-1))
        alpha_repeated = K.expand_dims(alpha) 
        weighted_input = inputs * alpha_repeated
        context_vector = K.sum(weighted_input, axis=1)
        return context_vector

    def get_config(self):
        return super(Attention, self).get_config()


# --- 2. Load and Preprocess Raw Signals ---
def load_and_preprocess_taps(file_list, sr):
    """Loads, truncates/pads, sums, and normalizes the three tap signals."""
    
    
    min_len = 44100 # Adjust this to match the minimum length used during your initial truncation step!
    
    signals = []
    
    for f in file_list:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return None
            
        # Load audio data
        y, s_r = librosa.load(f, sr=sr)
        
        # Truncate or Pad the signal to the minimum length
        if len(y) > min_len:
            y = y[:min_len]
        elif len(y) < min_len:
            pad_width = min_len - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
            
        signals.append(y.astype(np.float32))

    if not signals:
        return None
        
    # Summing Method: Element-wise addition of the three ridge signals
    combined_signal = np.sum(signals, axis=0)
    
    # Normalization (Must be the last step before MFCC extraction)
    normalized_combined_signal = librosa.util.normalize(combined_signal)
    
    return normalized_combined_signal


# --- 3. Feature Extraction and Reshaping ---
def extract_and_format_features(signal, n_mfcc, max_len, sr):
    """Extracts MFCCs and formats the array for the LSTM input shape."""
    
    # Extract MFCCs (40 x N_Frames)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    
    # Truncate/Pad the frames to MAX_MFCC_LENGTH
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    elif mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Reshape for LSTM/Attention Input: (1, Timesteps, Features) = (1, 100, 40)
    # Transpose from (Features, Timesteps) to (Timesteps, Features)
    mfccs_transposed = mfccs.T 
    
    # Add the batch dimension (1 sample)
    X_final = np.expand_dims(mfccs_transposed, axis=0)
    
    return X_final

# --- 4. Load Model and Predict ---
def predict_maturity(X_final):
    """Loads the trained model and performs inference."""
    
    try:
        # Load the model with the custom Attention layer
        model = load_model(MODEL_PATH, custom_objects={'Attention': Attention})
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and the Attention class is correctly defined.")
        return None

    # Prediction (returns an array of probabilities)
    probabilities = model.predict(X_final)
    
    # Find the index of the highest probability
    predicted_index = np.argmax(probabilities[0])
    
    # Map index to maturity label
    predicted_label = TARGET_NAMES.get(predicted_index, "Unknown")
    confidence = probabilities[0][predicted_index] * 100
    
    return predicted_label, confidence


# --- 5. EXECUTION ---
if __name__ == "__main__":
    
    print("--- Starting Coconut Maturity Prediction Pipeline ---")
    
    # A. Load and Preprocess Signals
    final_signal = load_and_preprocess_taps(WAV_FILES, SAMPLING_RATE)
    
    if final_signal is None:
        print("Failed to process input signals.")
    else:
        # B. Feature Extraction
        X_test_input = extract_and_format_features(final_signal, N_MFCC, MAX_MFCC_LENGTH, SAMPLING_RATE)
        print(f"MFCC Feature Input Shape: {X_test_input.shape} (Ready for Model)")
        
        # C. Prediction
        result = predict_maturity(X_test_input)
        
        if result:
            label, confidence = result
            print("\n-------------------------------------------")
            print(f"PREDICTION: {label}")
            print(f"CONFIDENCE: {confidence:.2f}%")
            print("-------------------------------------------")