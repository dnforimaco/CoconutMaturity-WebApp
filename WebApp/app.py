from flask import Flask, render_template
from backend import backend_bp

app = Flask(__name__)
app.register_blueprint(backend_bp)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect')
def detect_page():
    return render_template("detect.html")

@app.route('/about')
def about_page():
    return render_template("about.html")

@app.route('/history')
def history_page():
    return render_template("history.html")

if __name__ == "__main__":
    app.run(debug=True)
