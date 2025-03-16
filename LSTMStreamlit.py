import streamlit as st
import pandas as pd
import random
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from docx import Document



# Load our trained model 
cnn_model = load_model("CyberTrollModel_LSTM.keras")

# Load the same tokenizer used for the trained model
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)


# Define fun messages to choose one randomly after
fun_messages = {
    0: [
        "Oops! That message sounds like cyber trolling! 😈",
        "Uh-oh! This message might be a bit harsh. 😬",
        "Warning! This could be flagged as cyber trolling. 🚨",
        "Hmmm... Try rephrasing this kindly! 😊",
        "This message might hurt someone's feelings! 💔"
    ],
    1: [
        "Great! This message is respectful and friendly. 😊",
        "Nice! No signs of cyber trolling detected. 👍",
        "This message is safe and appropriate! 🎉",
        "You're spreading positivity! Keep it up! 🌟",
        "No issues detected! Keep the good vibes going! ✨"
    ]
}

# Function to preprocess text
def preprocess_text(text, max_length=100):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

# Function to predict using LSTM
def predict_text(text):
    processed_text = preprocess_text(text)
    pred = cnn_model.predict(processed_text)

    print(pred.shape, pred)  # Debugging output
    if pred.ndim == 2:
        pred = pred[0][0]
    elif pred.ndim == 1:
        pred = pred[0]
    
    pred = int(pred > 0.5)  # Seuil pour la classification binaire

    return pred, random.choice(fun_messages[pred])

# Function to process uploaded files
def process_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        df['Classification'] = df.iloc[:, 0].apply(lambda x: predict_text(x)[1])
        return df
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        texts = [para.text for para in doc.paragraphs if para.text.strip()]
        results = [predict_text(text)[1] for text in texts]
        return pd.DataFrame({"Text": texts, "Classification": results})
    else:
        return None

# Streamlit UI
st.title("CyberTroll Detection App - LSTM 🚀")
st.subheader("Check if a message is cyber trolling using LSTM!")

# Text input
user_text = st.text_area("Enter your message:")
if st.button("Check Message"):
    if user_text.strip():
        prediction, message = predict_text(user_text)
        st.success(f"{message}")
    else:
        st.warning("Please enter a valid message.")

# File upload section
st.subheader("Upload a File (CSV or Word) for Bulk Analysis")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "docx"])
if uploaded_file is not None:
    result_df = process_uploaded_file(uploaded_file)
    if result_df is not None:
        st.write(result_df)
        # Provide a download link
        result_df.to_csv("classified_results.csv", index=False)
        st.download_button("Download Classified File", "classified_results.csv")
    else:
        st.error("Unsupported file format. Please upload a CSV or Word document.")
