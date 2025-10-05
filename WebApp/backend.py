import os
from flask import Blueprint, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename

backend_bp = Blueprint("backend", __name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

        return render_template("detect.html", result=result)

    flash("Invalid file format. Please upload WAV/MP3/OGG")
    return redirect(url_for("detect_page"))
