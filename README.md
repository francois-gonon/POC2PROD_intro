# Stack Overflow Tag Prediction

A machine learning project that predicts Stack Overflow tags from post titles using BERT-based models.

## Project Overview

This project implements two-stage tag prediction:
- **Model 1**: Trained on primary tags (tag_position=0) to predict the most relevant tag
- **Model 2**: Trained on all tag positions to suggest multiple relevant tags

## Files

- `notebook.ipynb` - Training notebook with data preprocessing and model training
- `app.py` - Streamlit web application for tag prediction
- `stackoverflow_posts.csv` - Dataset with Stack Overflow posts and tags
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train models (optional - run notebook in Google Colab or locally):
```bash
# For Colab: Open notebook.ipynb in Google Colab
# For local: Run all cells in notebook.ipynb
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Training Models

The notebook performs the following steps:
1. Load and preprocess Stack Overflow post data
2. Remove rare labels and split data
3. Train Model 1 on primary tags (tag_position=0)
4. Train Model 2 on full dataset with all tag positions
5. Save both models to `model_stage1/` and `model_stage2/`

### Using the Web App

1. Launch the app with `streamlit run app.py`
2. Enter a Stack Overflow post title in the text box
3. View predictions from both models:
   - Model 1 shows the single most relevant tag
   - Model 2 shows top 3 suggested tags with probabilities

## Model Architecture

- Base model: `bert-base-uncased`
- Training: 3 epochs, batch size 16
- Text preprocessing: lowercase, remove URLs and special characters
- Maximum sequence length: 128 tokens

## Requirements

- Python 3.8+
- GPU recommended for training (not required for inference)
- ~2GB disk space for models

## Dataset

The dataset includes:
- `post_id` - Unique post identifier
- `tag_name` - Tag name
- `tag_id` - Tag identifier
- `tag_position` - Position of tag (0=primary)
- `title` - Post title text

## License

This is a student project for educational purposes.
