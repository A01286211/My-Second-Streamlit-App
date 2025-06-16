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
    
    st.subheader("Class Distribution")
    st.image("images/class_distribution.png", use_container_width=True)

    st.markdown("The Class Distribution histogram shows a fairly balanced dataset, with about a 1000 entries separating both labels")

    st.subheader("Token Length Histogram")
    st.image("images/token_lengths.png", use_container_width=True)

    st.markdown("The token length distribution chart, shows that the vast majority of entries are within the [0,1000] token length range.")

    st.subheader("Word Cloud")
    st.image("images/wordcloud.png", use_container_width=True)

    st.markdown("The word cloud for fake news exhibits the most common words found within the entries labeled as fake news. It allows for an ambiguous idea of what a fake news article may be about.")

    st.subheader("Noisy or Ambiguous Text Examples")
    st.markdown("Some example classified as ambiguous texts are:n\' [TITLE] aig quadruples limits for terrorism insurance to  billion [TEXT] part  religious fraud httpswwwyoutubecomwatchvqutkbhprbls'\n' [TITLE] no title [TEXT] in obamaland'\n' [TITLE] no title [TEXT] draining the swamp  youtubecomwatchvotakuaoi'\n' [TITLE] no title [TEXT] jamie gorelick call your office'\n' [TITLE] no title [TEXT] stolen but factual just shows how crooked they are'")
    st.markdown("It's noticed that most of these present links to some other site, or are of too little length to be of great training/test usage.")

elif page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")

    st.markdown("In this project, hyperparameter tuning was ultimately not performed due to a combination of practical constraints and encouraging initial results. One of the primary limitations was the runtime of the training pipeline. Even with access to Colab Pro, the training time for each model exceeded one hour, not even reaching the halfway point of a full training cycle. This made iterative experimentation using Optuna infeasible within the available computational power and time constraints. Additionally, the best-performing model, DistilBERT, achieved highly convincing classification accuracy using the default hyperparameters provided by the Hugging Face Trainer API. This suggests that the pretrained representations in transformer-based architectures already captured significant semantic and syntactic patterns relevant to the fake news detection task. In practical machine learning projects, if a model reaches strong baseline performance and the cost of tuning is disproportionately high, it is often reasonable to prioritize other improvements (e.g., better data quality or ensembling). Moreover, the dataset used posed inherent challenges such as label ambiguity and noisy text, meaning that further performance gains may depend more on data-centric strategies than on fine-tuning hyperparameters alone. Given the trade-offs in runtime, model performance, and resource availability, it was more productive to focus on evaluation, analysis, and model interpretability rather than deep tuning. Future work could definetly revisit hyperparameter optimization under more favorable conditions.")

elif page == "Model Analysis and Justification":
    st.title("Model Analysis and Justification")

    st.header("Model Justification")
    
    st.markdown("""
    Three transformer-based models were evaluated: **BERT**, **RoBERTa**, and **DistilBERT**. All are pretrained on large-scale text corpora and are well-established in NLP classification tasks.

    - **BERT** offers deep bidirectional context and is a solid baseline for many NLP problems.
    - **RoBERTa** improves on BERT with more training data, larger batches, and removal of the Next Sentence Prediction task.
    - **DistilBERT** is a smaller, faster version of BERT that retains most of its performance while being more efficient to train and deploy.

    Despite expectations, **DistilBERT significantly outperformed** both BERT and RoBERTa in this task. And it could be due to several factors. For instance, a lot of the entries classified as fake news were relatively short, or presented clear patterns like links to other sites, which could have favored this smaller model


    Overall, **DistilBERT was the best fit** for this project—offering a strong balance of speed, accuracy, and efficiency for real-world fake news detection.
    """)

    st.header("DistilBERT Performance Analysis")

    st.subheader("Classification Report")
    st.image("images/distilbert_classificationreport.png", use_container_width=True)
    
    st.subheader("Confusion Matrix")
    st.image("images/distilbert_confusionmatrix.png", use_container_width=True)

    st.markdown("The classification report for the DistilBERT model, presented a high accuracy and f1-score, with both being estimated at 0.98. Plus, the confusion matrix shows a very high number of True Positives and True Negatives, with only 158 total entries classified incorrectly.")

    st.header("BERT Performance Analysis")

    st.subheader("Classification Report")
    st.image("images/bert_classificationreport.png", use_container_width=True)
    
    st.subheader("Confusion Matrix")
    st.image("images/bert_confusionmatrix.png", use_container_width=True)

    st.markdown("The classification report for the BERT model, exposed the accuracy close to a random classifier achieved by this model, and the confusion matrix fully uncovered the problem, this model classified every entry as a fake news article. The latter, could've possibly been caused by using the same batch_size as the DistilBERT, as bigger models tend to need bigger batch sizes.")

    st.header("RoBERTa Performance Analysis")

    st.subheader("Classification Report")
    st.image("images/roberta_classificationreport.png", use_container_width=True)
    
    st.subheader("Confusion Matrix")
    st.image("images/roberta_confusionmatrix.png", use_container_width=True)

    st.markdown("RoBERTa presented the same symptoms as BERT, though this time, all entries were classified as real news articles. The cause for the problem could also be in the batch_size parameter, or maybe even in the limited epochs used for training")
