import os
import json
from datetime import datetime
from flask import Blueprint, request, redirect, url_for, render_template, flash, jsonify
from werkzeug.utils import secure_filename

backend_bp = Blueprint("backend", __name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}
HISTORY_FILE = "analysis_history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Analysis history storage
analysis_history = []

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_history():
    """Load analysis history from file"""
    global analysis_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                analysis_history = json.load(f)
                # Convert timestamp strings back to datetime objects
                for item in analysis_history:
                    if 'timestamp' in item and isinstance(item['timestamp'], str):
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
    except Exception as e:
        print(f"Error loading history: {e}")
        analysis_history = []

def save_history():
    """Save analysis history to file"""
    try:
        # Convert datetime objects to strings for JSON serialization
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
    """Get recent analyses for display on detect page"""
    # Sort by timestamp (newest first)
    sorted_history = sorted(analysis_history, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
    return sorted_history[:limit]

# Load history on startup
load_history()

@backend_bp.route("/classify", methods=["POST"])
def classify():
    if "audio" not in request.files:
        flash("No file uploaded!")
        return redirect(url_for("detect_page"))

    file = request.files["audio"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("detect_page"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 🔗 Connect your ML model here
        # from model import classify_audio
        # result = classify_audio(filepath)
        result = "Mature Coconut"  # temporary placeholder

        # Store analysis in history
        analysis_entry = {
            'id': len(analysis_history) + 1,
            'filename': filename,
            'result': result,
            'confidence': 85,  # This would come from your ML model
            'timestamp': datetime.now(),
            'filepath': filepath
        }
        analysis_history.append(analysis_entry)
        save_history()

        return render_template("detect.html", result=result)

    flash("Invalid file format. Please upload WAV/MP3/OGG")
    return redirect(url_for("detect_page"))

@backend_bp.route("/api/history", methods=["GET"])
def get_history():
    """Get all analysis history"""
    return jsonify(analysis_history)

@backend_bp.route("/api/history", methods=["POST"])
def sync_analysis():
    """Sync analysis data from client to server"""
    try:
        analysis_data = request.get_json()
        if not analysis_data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        # Add to server history if not already exists
        existing_ids = [item['id'] for item in analysis_history]
        if analysis_data.get('id') not in existing_ids:
            # Convert timestamp string back to datetime if needed
            if 'timestamp' in analysis_data and isinstance(analysis_data['timestamp'], str):
                analysis_data['timestamp'] = datetime.fromisoformat(analysis_data['timestamp'])

            analysis_history.append(analysis_data)
            save_history()

        return jsonify({'success': True, 'message': 'Analysis synced successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@backend_bp.route("/api/history/recent", methods=["GET"])
def get_recent_history():
    """Get recent analyses for detect page"""
    limit = request.args.get('limit', 5, type=int)
    recent = get_recent_analyses(limit)
    return jsonify(recent)

@backend_bp.route("/api/history/<int:item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    """Delete a specific analysis from history"""
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
    """Clear all analysis history"""
    global analysis_history
    analysis_history = []
    save_history()
    return jsonify({'success': True, 'message': 'All analyses cleared successfully'})
