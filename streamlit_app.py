import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.express as px

# Load models and tokenizer
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_2 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_3 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    model_1.load_state_dict(torch.load("model_1.pth", map_location=torch.device('cpu')))
    model_2.load_state_dict(torch.load("model_2.pth", map_location=torch.device('cpu')))
    model_3.load_state_dict(torch.load("model_3.pth", map_location=torch.device('cpu')))

    model_1.eval()
    model_2.eval()
    model_3.eval()

    return tokenizer, model_1, model_2, model_3

tokenizer, model_1, model_2, model_3 = load_models()

def predict(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        prediction = torch.argmax(probs).item()
    return prediction, probs

st.set_page_config(page_title="Fake News Classifier", layout="wide")

# Sidebar for navigation
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
    selected_model = st.selectbox("Choose a model:", ["Model 1", "Model 2", "Model 3"])

    if st.button("Classify") and user_input:
        model = {"Model 1": model_1, "Model 2": model_2, "Model 3": model_3}[selected_model]
        label, scores = predict(user_input, model)
        st.markdown(f"**Prediction:** {'Fake News' if label == 1 else 'Real News'}")
        st.markdown("**Confidence Scores:**")
        st.json({"Real News": float(scores[0]), "Fake News": float(scores[1])})

# Page 2: Dataset Visualization
elif page == "Dataset Visualization":
    st.title("Dataset Visualization")
    # Replace with actual dataset
    # df = pd.read_csv('your_dataset.csv')
    # Plot class distribution, token lengths, word cloud, etc.
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
    # Include Optuna or Keras Tuner visualizations or screenshots here
    # Example:
    # st.image("optuna_study_plot.png")

    st.markdown("## Tuned Parameters")
    # st.write({"learning_rate": 2e-5, "batch_size": 32, "dropout_rate": 0.3})

    st.markdown("## Performance Over Trials")
    # Include a line plot of performance (e.g., F1-score) across trials

# Page 4: Model Analysis and Justification
elif page == "Model Analysis and Justification":
    st.title("Model Analysis and Justification")
    st.markdown("### Dataset Challenges")
    # Write about class imbalance, multilinguality, etc.

    st.markdown("### Prior Work")
    # Cite relevant papers or Kaggle solutions

    st.markdown("### Why Our Model?")
    # Justify choice of model architecture

    st.markdown("### Classification Report")
    # from sklearn.metrics import classification_report
    # y_true, y_pred = ...
    # report = classification_report(y_true, y_pred, output_dict=True)
    # st.json(report)

    st.markdown("### Confusion Matrix")
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_true, y_pred)
    # fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    # st.pyplot(fig)

    st.markdown("### Error Analysis")
    # st.write("False Positives and False Negatives Examples")
    # st.write(df[(y_pred != y_true)].head())
    # Discuss patterns and suggestions for improvement
