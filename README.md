# Sentiment Analysis with Cohere Embeddings 

An interactive sentiment analysis application that determines emotions from text input using Cohere's multilingual embeddings and machine learning classification.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-API-00D9FF?style=for-the-badge)

## Overview

Emotional expression is subjective and shaped by cultural background, emotional intelligence, and individual personality traits. While we can communicate emotions clearly in verbal communication due to nuances of tone and inflection, it's harder to determine how a person feels through text alone.

This application uses sentiment analysis powered by **Cohere's embed-multilingual-v3.0** model to determine emotions from text input across **8 different emotion categories**: Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, and Trust.

## Features

- **Multilingual Support**: Analyzes text in 100+ languages using Cohere's multilingual embeddings
- **8 Emotion Categories**: Detects Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, and Trust
- **Multi-label Classification**: Can identify multiple emotions in a single text
- **Interactive UI**: Beautiful Streamlit interface with emotion GIFs
- **Real-time Analysis**: Instant emotion prediction as you type
- **Probability Scores**: Shows confidence levels for each detected emotion

## 🏗️ Architecture

### Step-by-Step Process

1. **Gather Emotion Data**: Uses the XED dataset with 8 emotion categories
2. **Train Emotion Classifier**: Trains a classifier chain with logistic regression
3. **Get User Input**: Captures text via Streamlit interface
4. **Embed Input**: Generates embeddings using Cohere's embed-multilingual-v3.0
5. **Classify Sentiment**: Predicts emotion probabilities using trained classifier
6. **Display Results**: Shows top emotions with GIFs and probability scores

### Technical Stack

- **Embeddings**: Cohere embed-multilingual-v3.0 model
- **Classification**: Scikit-learn's ClassifierChain with LogisticRegression
- **Dataset**: XED (Cross-lingual Emotion Dataset)
- **Frontend**: Streamlit
- **ML Framework**: PyTorch, Scikit-learn

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Cohere API key ([Get one free here](https://dashboard.cohere.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/co-sentiment.git
   cd co-sentiment
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Cohere API key**
   ```bash
   export COHERE_API_KEY='your-api-key-here'
   ```
   
   Or create a `.env` file:
   ```
   COHERE_API_KEY=your-api-key-here
   ```

### Running the Application

```bash
streamlit run sentiment.py
```

The app will open in your browser at `http://localhost:8501`

For headless mode (useful for servers):
```bash
streamlit run sentiment.py --server.headless=true --server.port=8501
```

## Usage

1. **Enter your text**: Type how you're feeling in the text input box
2. **Select top emotions**: Use the slider to choose how many emotions to display (1-8)
3. **View results**: See the detected emotions with animated GIFs and probability percentages

### Example Inputs

- "I can't wait for tomorrow's surprise!" → Anticipation, Joy
- "I'm worried about the upcoming exam" → Fear, Sadness
- "This is absolutely disgusting!" → Disgust, Anger
- "I feel so grateful for my friends" → Trust, Joy

## Model Training

The classifier is automatically trained on first run if the model file doesn't exist. The training process:

1. Loads pre-computed embeddings from `xed_with_embeddings.json`
2. Applies MultiLabelBinarizer for one-hot encoding
3. Trains a ClassifierChain with LogisticRegression
4. Saves the trained model to `data/models/emotion_chain.pkl`

To manually retrain the model, delete `data/models/emotion_chain.pkl` and restart the app.

## Project Structure

```
co-sentiment/
├── data/
│   ├── emotions/           # Emotion GIF files
│   │   ├── anger.gif
│   │   ├── anticipation.gif
│   │   ├── disgust.gif
│   │   ├── fear.gif
│   │   ├── joy.gif
│   │   ├── sadness.gif
│   │   ├── surprise.gif
│   │   └── trust.gif
│   ├── models/             # Trained model storage
│   │   └── emotion_chain.pkl
│   └── xed_with_embeddings.json  # Pre-computed embeddings
├── sentiment.py            # Main application file
├── utils.py               # Utility functions (embeddings, etc.)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## Configuration

### Model Parameters

In `sentiment.py`, you can configure:

```python
model_name = 'embed-multilingual-v3.0'  # Cohere embedding model
test_size = 0.01                        # Train/test split ratio
random_state = 42                       # Reproducibility seed
```

### Embedding Parameters

In `utils.py`, customize embedding generation:

```python
input_type = "search_document"   # Embedding input type
truncate = 'RIGHT'               # Text truncation strategy
batch_size = 2048                # Batch size for API calls
```
