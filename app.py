import streamlit as st
import torch
import json
import re
from transformers import BertTokenizer, BertForSequenceClassification

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load models
@st.cache_resource
def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    with open(f'{model_path}/label_dict.json', 'r') as f:
        label_dict = json.load(f)
    id_to_label = {v: k for k, v in label_dict.items()}
    return model, tokenizer, id_to_label

model_stage1, tokenizer_stage1, id_to_label_stage1 = load_model('model_stage1')
model_stage2, tokenizer_stage2, id_to_label_stage2 = load_model('model_stage2')

# Prediction function for single tag
def predict_single(text, model, tokenizer, id_to_label):
    clean_text = preprocess_text(text)
    encoding = tokenizer(clean_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        pred_id = outputs.logits.argmax(dim=1).item()
    return id_to_label[str(pred_id)]

# Prediction function for multiple tags
def predict_multiple(text, model, tokenizer, id_to_label, top_k=3):
    clean_text = preprocess_text(text)
    encoding = tokenizer(clean_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        top_probs, top_ids = torch.topk(probs, top_k, dim=1)
    results = [(id_to_label[str(idx.item())], prob.item()) for idx, prob in zip(top_ids[0], top_probs[0])]
    return results

# Streamlit interface
st.title('Stack Overflow Tag Prediction')
st.write('Enter a Stack Overflow post title to predict its tag')

user_input = st.text_input('Post Title:', placeholder='e.g., How to reverse a list in Python?')

if user_input:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Model Stage 1')
        st.caption('Trained on tag_position = 0 only')
        pred1 = predict_single(user_input, model_stage1, tokenizer_stage1, id_to_label_stage1)
        st.success(f'Most relevant tag: **{pred1}**')
    
    with col2:
        st.subheader('Model Stage 2')
        st.caption('Trained on full dataset')
        preds2 = predict_multiple(user_input, model_stage2, tokenizer_stage2, id_to_label_stage2, top_k=3)
        st.success('**Suggested tags:**')
        for tag, prob in preds2:
            st.write(f'- {tag} ({prob:.2%})')
