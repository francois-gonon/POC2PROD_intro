import streamlit as st
import torch
import json
import re
import pandas as pd

# Clean text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load models once
@st.cache_resource
def load_models():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Load tag names once
    df = pd.read_csv('stackoverflow_posts.csv')
    tag_id_to_name = dict(zip(df['tag_id'], df['tag_name']))
    
    # Model 1
    model1 = AutoModelForSequenceClassification.from_pretrained('model_stage1')
    tokenizer1 = AutoTokenizer.from_pretrained('model_stage1')
    with open('model_stage1/label_dict.json', 'r') as f:
        label_dict1 = json.load(f)
        labels1 = {v: tag_id_to_name.get(int(k), f'tag_{k}') for k, v in label_dict1.items()}
    
    # Model 2
    model2 = AutoModelForSequenceClassification.from_pretrained('model_stage2')
    tokenizer2 = AutoTokenizer.from_pretrained('model_stage2')
    with open('model_stage2/label_dict.json', 'r') as f:
        label_dict2 = json.load(f)
        labels2 = {v: tag_id_to_name.get(int(k), f'tag_{k}') for k, v in label_dict2.items()}
    
    return model1, tokenizer1, labels1, model2, tokenizer2, labels2

model1, tokenizer1, labels1, model2, tokenizer2, labels2 = load_models()

# Get predictions
def predict(text, model, tokenizer, labels, top_k=3):
    clean = preprocess_text(text)
    inputs = tokenizer(clean, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        top_probs, top_ids = torch.topk(probs, top_k, dim=1)
    return [(labels.get(idx.item(), 'unknown'), prob.item()) for idx, prob in zip(top_ids[0], top_probs[0])]

st.title('Stack Overflow Tag Predictor')

title = st.text_input('Enter post title:')

if title:
    # Run both models
    pred1 = predict(title, model1, tokenizer1, labels1, top_k=1)
    preds2 = predict(title, model2, tokenizer2, labels2, top_k=3)
    
    # Display results side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**Model 1 (position=0 only)**')
        st.write(f'**{pred1[0][0]}** ({pred1[0][1]:.1%})')
    
    with col2:
        st.write('**Model 2 (all positions)**')
        for tag, prob in preds2:
            st.write(f'{tag}: {prob:.1%}')
