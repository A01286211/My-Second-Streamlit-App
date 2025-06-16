import streamlit as st
import os
import zipfile
import torch
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Google Drive file IDs
GDRIVE_FILE_IDS = {
    "DistilBERT": "1xwm50FzQJFrxZovV1AePsVOwtvugzFn7",
    "BERT": "1hEUnQihsGjjtfkCqEfxjHAvujNLt-PTv",     
    "RoBERTa": "1AezznK-va0QZ2hRrZydaRxNIWuvp6CY6"   
}

def download_and_extract_from_drive(file_id, extract_path, zip_name):
    os.makedirs(extract_path, exist_ok=True)
    zip_path = os.path.join("/tmp", zip_name)

    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    return extract_path

@st.cache_resource
def load_models():
    models = {}
    for name, file_id in GDRIVE_FILE_IDS.items():
        extract_path = f"/tmp/{name.lower()}_model"
        model_path = download_and_extract_from_drive(file_id, extract_path, f"{name}.zip")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        models[name] = (tokenizer, model)
    return models

# Prediction function
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        prediction = torch.argmax(probs).item()
    return prediction, probs

st.set_page_config(page_title="Fake News Classifier", layout="wide")
page = st.sidebar.selectbox("Choose a page:", [
    "Inference Interface",
    "Dataset Visualization",
    "Hyperparameter Tuning",
    "Model Analysis and Justification"
])

models = load_models()

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

elif page == "Dataset Visualization":
    st.title("Dataset Visualization")
    # COMMENT: Insert class distribution, token lengths, wordcloud, etc.

elif page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    # COMMENT: Show visualizations or screenshots of tuning results

elif page == "Model Analysis and Justification":
    st.title("Model Analysis and Justification")
    # COMMENT: Insert classification report, confusion matrix, and error analysis
