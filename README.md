
# UniFeedback AI 🎓

### University Feedback Intelligence Platform

UniFeedback AI is a standalone Generative AI
application designed to analyze university student
feedback and generate actionable insights.

## Project Overview

The application uses Qwen 2.5 3B through the
local Ollama REST API to process student reviews.

It helps universities identify student concerns,
understand feedback trends, and improve educational
quality.

## Key Features

- Upload CSV and Excel files.
- Clean and preprocess student feedback.
- Sentiment classification:
  Positive, Negative, Neutral, and Mixed.
- Automatic topic extraction and grouping.
- Identification of main issues and positive aspects.
- AI-generated recommendations.
- Executive summary generation.
- Interactive analytics dashboard.
- Temperature and Max Output Tokens controls.
- Error handling and evaluation tools.
- Export analysis results.

## Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Ollama REST API
- Qwen 2.5 3B
- Scikit-learn

## Prompt Engineering

The application uses:
- Role Prompting
- Few-Shot Prompting

## Installation

### 1. Install Python

Install Python and Git.

### 2. Install Ollama

Download Ollama from:

https://ollama.com/

### 3. Download the AI Model

Run:

ollama pull qwen2.5:3b

### 4. Clone the Repository

git clone YOUR_REPOSITORY_URL

cd UniFeedback-AI

### 5. Create a Virtual Environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

### 6. Install Dependencies

pip install -r requirements.txt

### 7. Run the Application

streamlit run app.py

Open the local URL displayed in the terminal.

## Model Evaluation

The model achieved 98% accuracy on a
synthetic test dataset of 50 reviews.

This result applies only to the tested dataset
and does not guarantee the same performance
on unseen real-world feedback.

## Model Configuration

- Model: Qwen 2.5 3B
- API: Local Ollama REST API
- Temperature: Configurable
- Maximum Output Tokens: Configurable

## Privacy

The model runs locally through Ollama.
Do not upload confidential or personally
identifiable student feedback.

## Author

Sara Hany Aly
