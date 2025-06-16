# 📰 Fake News Classifier using Transformer Models

This repository contains a full Streamlit web application that classifies news text as either **real** or **fake**, using fine-tuned transformer-based models. Users can interact with three different models (DistilBERT, BERT, and RoBERTa), explore dataset visualizations, and review detailed model analysis.

---

## 🚀 Live Demo

Try the app live on Streamlit Cloud:  
👉 **[https://my-second-app-szd.streamlit.app/](https://my-second-app-szd.streamlit.app/)**

---

## 📁 Repository

GitHub Repository:  
👉 **[https://github.com/A01286211/My-Second-Streamlit-App](https://github.com/A01286211/My-Second-Streamlit-App)**

---

## 🧠 Models

The app uses three transformer models:

- `DistilBERT`
- `BERT`
- `RoBERTa`

They were fine-tuned on a fake news classification dataset but are too large to store in the repo. Instead, they are hosted externally:

📦 **Download Models from Google Drive:**  
👉 [https://drive.google.com/drive/folders/1yxKA8RkMeU7PVfer-5eZ95Oqhm7qNKqn?usp=drive_link](https://drive.google.com/drive/folders/1yxKA8RkMeU7PVfer-5eZ95Oqhm7qNKqn?usp=drive_link)

On first run, the app downloads and unzips these models automatically.

---

## 🖼️ App Structure (Streamlit Pages)

### 1. **Inference Interface**
- Input any text
- Select one of the three models
- View prediction (real or fake) and model confidence

### 2. **Dataset Visualization**
- Visual insights: class distribution, token length histograms, and word clouds
- Screenshots generated offline and displayed in the app

### 3. **Hyperparameter Tuning**
- ❌ No tuning was done due to:
  - Runtime limitations (training took over 1 hour per epoch)
  - DistilBERT already achieved strong performance without tuning
  - Prioritization of efficiency over marginal gains

### 4. **Model Analysis & Justification**
- Classification report and confusion matrix
- Error analysis on false positives/negatives
- DistilBERT outperformed due to:
  - Smaller size and faster training
  - Better generalization on limited data
  - Compatibility with Colab Pro constraints

---

## 🛠️ Installation (Run Locally)

```bash
# Clone the repo
git clone https://github.com/A01286211/My-Second-Streamlit-App.git
cd My-Second-Streamlit-App

# (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

```
---

## 🙌 Built With

- 🤗 Hugging Face Transformers  
- 🔥 PyTorch  
- 🧼 Streamlit  
- 📊 Scikit-learn  
- 📈 Seaborn, Plotly, Matplotlib  
- 💻 Google Colab Pro (for training)

---

## 👤 Author

Developed by **[A01286211](https://github.com/A01286211)**  
For educational and demonstration purposes.

---
