import os
import json
import sys
from datetime import datetime
from flask import Blueprint, request, redirect, url_for, render_template, flash, jsonify
from werkzeug.utils import secure_filename

# --- ML IMPORTS (UNCOMMENTED AND NECESSARY) ---
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

backend_bp = Blueprint("backend", __name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}
HISTORY_FILE = "analysis_history.json"
# Ensure your model file is in the correct path relative to your application
MODEL_PATH = "model/attention_lstm_model.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- ML Model Configuration (Confirmed) ---
SAMPLING_RATE = 44100
N_MFCC = 40
# This is the fixed number of frames/timesteps you used for padding/truncation
MAX_MFCC_LENGTH = 100 
NUM_CLASSES = 3
TARGET_NAMES = {0: 'Premature', 1: 'Mature', 2: 'Overmature'}

# --- Custom Attention Layer (UNCOMMENTED) ---
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
    
# Global variable to hold the loaded model
CLASSIFICATION_MODEL = None

# --- Analysis history functions (Unchanged) ---
analysis_history = []

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_history():
    # ... (History loading logic is unchanged)
    global analysis_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                analysis_history = json.load(f)
                for item in analysis_history:
                    if 'timestamp' in item and isinstance(item['timestamp'], str):
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
    except Exception as e:
        print(f"Error loading history: {e}")
        analysis_history = []

def save_history():
    # ... (History saving logic is unchanged)
    try:
        history_to_save = []
        for item in analysis_history:
            item_copy = item.copy()
            if 'timestamp' in item_copy and isinstance(item_copy['timestamp'], datetime):
                item_copy['timestamp'] = item_copy['timestamp'].isoformat()
            history_to_save.append(item_copy)

        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_to_save, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def get_recent_analyses(limit=5):
    # ... (Recent analysis logic is unchanged)
    sorted_history = sorted(analysis_history, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
    return sorted_history[:limit]

# Load history on startup
load_history()

# --- ML Model Functions (IMPLEMENTED) ---
def load_and_preprocess_taps(file_list, sr):
    """Loads, truncates/pads, sums, and normalizes the three tap signals."""
    
    # Assume the minimum required sample length for the raw audio is 44100 (1 second)
    # This must match the truncation length used during training!
    min_len = 44100 
    
    signals = []
    
    for f in file_list:
        try:
            # Load audio data
            y, s_r = librosa.load(f, sr=sr)
        except Exception as e:
            print(f"Librosa error loading {f}: {e}")
            return None

        # Truncate or Pad the signal to the minimum length
        if len(y) > min_len:
            y = y[:min_len]
        elif len(y) < min_len:
            pad_width = min_len - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
            
        signals.append(y.astype(np.float32))

    # Summing Method: Element-wise addition of the three ridge signals
    combined_signal = np.sum(signals, axis=0)
    
    # Normalization
    normalized_combined_signal = librosa.util.normalize(combined_signal)
    
    return normalized_combined_signal

def extract_and_format_features(signal, n_mfcc, max_len, sr):
    """Extracts MFCCs and formats the array for the LSTM input shape (1, 100, 40)."""
    
    # Extract MFCCs (40 x N_Frames)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    
    # Truncate/Pad the frames to MAX_MFCC_LENGTH (100)
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    elif mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Transpose from (Features, Timesteps) to (Timesteps, Features)
    mfccs_transposed = mfccs.T 
    
    # Add the batch dimension (1 sample) -> (1, 100, 40)
    X_final = np.expand_dims(mfccs_transposed, axis=0)
    
    return X_final

def load_classification_model():
    """Load the model once on application startup."""
    global CLASSIFICATION_MODEL
    if CLASSIFICATION_MODEL is None:
        try:
            print(f"Attempting to load model from: {MODEL_PATH}")
            # Load the model with the custom Attention layer
            CLASSIFICATION_MODEL = load_model(
                MODEL_PATH, 
                custom_objects={'Attention': Attention}
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            CLASSIFICATION_MODEL = None
            
    return CLASSIFICATION_MODEL


def predict_maturity(X_final):
    """Performs inference using the loaded model."""
    
    model = load_classification_model()
    if model is None:
        return None, None, "Model failed to load on startup."

    # Prediction (returns an array of probabilities, e.g., [[0.05, 0.94, 0.01]])
    probabilities = model.predict(X_final)
    
    # Find the index of the highest probability
    predicted_index = np.argmax(probabilities[0])
    
    # Map index to maturity label
    predicted_label = TARGET_NAMES.get(predicted_index, "Unknown")
    confidence = probabilities[0][predicted_index] * 100
    
    return predicted_label, confidence, None


def classify_coconut_audio(file_paths):
    """Main function to classify coconut maturity from three tap recordings."""

    if len(file_paths) != 3:
        return None, None, "Please provide exactly 3 audio files"

    # 1. Load and Preprocess Signals (Summing Method)
    final_signal = load_and_preprocess_taps(file_paths, SAMPLING_RATE)
    if final_signal is None:
        return None, None, "Failed during audio loading or preprocessing."

    # 2. Feature Extraction (MFCC)
    X_test_input = extract_and_format_features(final_signal, N_MFCC, MAX_MFCC_LENGTH, SAMPLING_RATE)
    
    # 3. Predict Maturity using the ML Model
    result, confidence, error_message = predict_maturity(X_test_input)

    return result, confidence, error_message

# Load model on application start (optional, but good practice)
load_classification_model()


# --- Flask Routes (Modified to use real ML function) ---

@backend_bp.route("/classify", methods=["POST"])
def classify():
    """Handle multiple file uploads for coconut maturity classification"""

    uploaded_files = request.files.getlist("audio_files")
    file_names = []

    if len(uploaded_files) != 3:
        flash("Please upload exactly 3 audio files for analysis")
        return redirect(url_for("detect_page"))

    valid_files = []
    for file in uploaded_files:
        if file.filename == "":
            flash("One or more files are missing")
            return redirect(url_for("detect_page"))

        if file and allowed_file(file.filename):
            # Use a timestamp or UUID in the filename to prevent overwriting during concurrent uploads
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            original_filename = secure_filename(file.filename)
            unique_filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)
            valid_files.append(filepath)
            file_names.append(original_filename) # Keep original name for history display
        else:
            flash("Invalid file format. Please upload WAV/MP3/OGG files only")
            return redirect(url_for("detect_page"))

    if len(valid_files) != 3:
        flash("All files must be valid audio files")
        return redirect(url_for("detect_page"))

    # Classify using the REAL ML model function
    result, confidence, error_message = classify_coconut_audio(valid_files)

    if error_message:
        flash(f"Classification error: {error_message}")
        return redirect(url_for("detect_page"))

    # Create a combined filename for display
    combined_filename = " + ".join(file_names)

    # Store analysis in history
    analysis_entry = {
        'id': len(analysis_history) + 1,
        'filename': combined_filename,
        'files': file_names,
        'result': result,
        'confidence': round(confidence, 2),
        'timestamp': datetime.now(),
        'filepaths': valid_files
    }
    analysis_history.append(analysis_entry)
    save_history()

    # Pass results to the template
    return render_template("detect.html", result=result, confidence=round(confidence, 2), recent_analyses=get_recent_analyses())

# --- Other Flask Routes (Unchanged) ---

@backend_bp.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(analysis_history)

@backend_bp.route("/api/history", methods=["POST"])
def sync_analysis():
    # ... (Sync logic is unchanged)
    try:
        analysis_data = request.get_json()
        if not analysis_data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        existing_ids = [item['id'] for item in analysis_history]
        if analysis_data.get('id') not in existing_ids:
            if 'timestamp' in analysis_data and isinstance(analysis_data['timestamp'], str):
                analysis_data['timestamp'] = datetime.fromisoformat(analysis_data['timestamp'])

            analysis_history.append(analysis_data)
            save_history()

        return jsonify({'success': True, 'message': 'Analysis synced successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@backend_bp.route("/api/history/recent", methods=["GET"])
def get_recent_history():
    limit = request.args.get('limit', 5, type=int)
    recent = get_recent_analyses(limit)
    return jsonify(recent)

@backend_bp.route("/api/history/<int:item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    global analysis_history
    original_length = len(analysis_history)
    analysis_history = [item for item in analysis_history if item['id'] != item_id]

    if len(analysis_history) < original_length:
        save_history()
        return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Analysis not found'}), 404

@backend_bp.route("/api/history", methods=["DELETE"])
def clear_all_history():
    global analysis_history
    analysis_history = []
    save_history()
    return jsonify({'success': True, 'message': 'All analyses cleared successfully'})

if __name__ == '__main__':
    # This block is typically for testing the backend module directly
    # In a real Flask app, this would run via a main app file
    load_classification_model()
