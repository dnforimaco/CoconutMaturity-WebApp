# Coconut Maturity Detection Web App

This web application uses Artificial Intelligence to classify the maturity stage of coconuts based on audio data from tapping sounds. It helps farmers determine if coconuts are Premature, Mature, or Overmature, enabling better harvesting decisions and reducing waste.

## Features

- **AI-Powered Classification**: Utilizes a TensorFlow model with Attention LSTM for accurate coconut maturity prediction
- **Web Interface**: User-friendly Flask-based web application for uploading audio files and viewing results
- **Audio Processing**: Advanced audio feature extraction using MFCCs (Mel-Frequency Cepstral Coefficients) and Librosa
- **History Tracking**: Maintains a record of previous classifications for analysis
- **Responsive Design**: Modern UI with responsive design for desktop and mobile use

## Technology Stack

- **Backend**: Python with Flask
- **ML Framework**: TensorFlow with custom Attention layer
- **Audio Processing**: Librosa, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with modern design patterns

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/coconut-maturity-detection.git
cd coconut-maturity-detection
```

### Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
coconut-maturity-detection/
├── app.py                      # Main Flask application
├── backend.py                  # Backend logic and routes
├── classification_code.py      # ML model and prediction logic
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── analysis_history.json       # Classification history storage
├── model/                      # Trained ML model
│   └── attention_lstm_model.h5
├── static/                     # Static files (CSS, JS, images)
│   └── css/
│       └── style.css
└── templates/                  # HTML templates
    ├── base.html
    ├── index.html
    ├── detect.html
    ├── history.html
    └── about.html
```

## Running the Application

### Start the Development Server

```bash
python app.py
```

The application will run on `http://localhost:5000` by default.

### Access the Application

Open your web browser and navigate to: `http://localhost:5000`

## Usage

1. **Homepage**: Learn about coconut maturity stages and farming challenges
2. **Detection**: Upload audio files (three tap recordings recommended) for maturity classification
3. **History**: View previous classification results and analysis
4. **About**: Learn more about the project and technology

## How It Works

1. **Audio Upload**: Users upload three audio files representing coconut tapping sounds
2. **Preprocessing**: Audio signals are normalized, truncated/padded, and combined
3. **Feature Extraction**: MFCC features are extracted using Librosa
4. **Prediction**: Features are fed into the trained Attention LSTM model
5. **Results**: The model outputs maturity classification (Premature/Mature/Overmature) with confidence score

## API Endpoints

- `GET /` - Homepage
- `GET /detect` - Detection page
- `GET /about` - About page
- `GET /history` - History page
- `POST /classify` - Classification endpoint (via backend.py)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and Keras for deep learning framework
- Librosa for audio processing capabilities
- Flask for web framework
- Coconut farmers and researchers for domain knowledge
- University of the Philippines Los Baños (sample data sources)

## Future Improvements

- Mobile app development
- Real-time audio recording and classification
- Integration with farming management systems
- Expanded dataset for better model accuracy
- Hardware prototype for field deployment

## Support

For questions, issues, or contributions, please contact the project maintainers or create an issue in the repository.
