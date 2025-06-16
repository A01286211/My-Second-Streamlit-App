import streamlit as st
import zipfile
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.express as px

# Paths
MODEL_ZIP_PATHS = {
    "DistilBERT": "models/distilbert_model.zip",
    "BERT": "models/bert.zip",
    "RoBERTa": "models/roberta.zip"
}
MODEL_EXTRACT_PATHS = {
    "DistilBERT": "/tmp/distilbert_model",
    "BERT": "/tmp/bert_model",
    "RoBERTa": "/tmp/roberta_model"
}

# Extract and load models
@st.cache_resource
def load_models():
    models = {}
    for name, zip_path in MODEL_ZIP_PATHS.items():
        extract_path = MODEL_EXTRACT_PATHS[name]

        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        tokenizer = AutoTokenizer.from_pretrained(extract_path)
        model = AutoModelForSequenceClassification.from_pretrained(extract_path)
        model.eval()
        models[name] = (tokenizer, model)
    return models

models = load_models()

# Prediction function
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        prediction = torch.argmax(probs).item()
    return prediction, probs

st.set_page_config(page_title="Fake News Classifier", layout="wide")

# Navigation
page = st.sidebar.selectbox("Choose a page:", [
    "Inference Interface",
    "Dataset Visualization",
    "Hyperparameter Tuning",
    "Model Analysis and Justification"
])

# Page 1: Inference Interface
if page == "Inference Interface":
    st.title("Fake News Detection")
    user_input = st.text_area("Enter text to classify:")
    selected_model = st.selectbox("Choose a model:", list(models.keys()))

    if st.button("Classify") and user_input:
        tokenizer, model = models[selected_model]
        label, scores = predict(user_input, model, tokenizer)
        st.markdown(f"**Prediction:** {'Fake News' if label == 1 else 'Real News'}")
        st.markdown("**Confidence Scores:**")
        st.json({"Real News": float(scores[0]), "Fake News": float(scores[1])})

# Page 2: Dataset Visualization
elif page == "Dataset Visualization":
    st.title("Dataset Visualization")
    # COMMENT: Replace below with actual dataset loading and visualization
    st.markdown("## Class Distribution")
    # st.bar_chart(df['label'].value_counts())

    st.markdown("## Token Length Histogram")
    # token_lengths = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
    # fig, ax = plt.subplots()
    # ax.hist(token_lengths, bins=50)
    # st.pyplot(fig)

    st.markdown("## Word Cloud")
    # text = ' '.join(df['text'])
    # wordcloud = WordCloud(width=800, height=400).generate(text)
    # fig, ax = plt.subplots()
    # ax.imshow(wordcloud, interpolation='bilinear')
    # ax.axis("off")
    # st.pyplot(fig)

    st.markdown("## Noisy or Ambiguous Text Examples")
    # st.write(df[df['label'] == 'ambiguous'].head())

# Page 3: Hyperparameter Tuning
elif page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    st.markdown("## Optimization Process")
    # COMMENT: Insert visualizations or screenshots from Optuna/KerasTuner
    # st.image("images/optuna_plot.png")

    st.markdown("## Tuned Parameters")
    # st.write({\"learning_rate\": 2e-5, \"batch_size\": 32, \"dropout_rate\": 0.3})

    st.markdown("## Performance Over Trials")
    # COMMENT: Line plot showing F1-score over tuning steps

# Page 4: Model Analysis and Justification
elif page == "Model Analysis and Justification":
    st.title("Model Analysis and Justification")
    st.markdown("### Dataset Challenges")
    # COMMENT: Discuss imbalance, noise, multilinguality, etc.

    st.markdown("### Prior Work")
    # COMMENT: Cite relevant research papers or Kaggle solutions

    st.markdown("### Why Our Model?")
    # COMMENT: Justify architecture choice like DistilBERT, RoBERTa, etc.

    st.markdown("### Classification Report")
    # COMMENT: Insert classification report via sklearn

    st.markdown("### Confusion Matrix")
    # COMMENT: Include confusion matrix visualization

    st.markdown("### Error Analysis")
    # COMMENT: Show false positives/negatives and improvement suggestions
