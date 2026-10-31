"""Simple web interface for the SMS spam detector.
Run (after training):  streamlit run app.py
"""
import json, pickle, re
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\b\d{5,}\b", " numtoken ", t)
    t = re.sub(r"[^a-z0-9\s£$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

@st.cache_resource
def load():
    model = tf.keras.models.load_model("models/spam_bilstm.keras")
    tok = pickle.load(open("models/tokenizer.pkl", "rb"))
    max_len = json.load(open("models/config.json"))["MAX_LEN"]
    return model, tok, max_len

st.title("SMS Spam Detector (BiLSTM)")
msg = st.text_area("Enter an SMS message:", height=120)
if st.button("Check message") and msg.strip():
    model, tok, max_len = load()
    seq = pad_sequences(tok.texts_to_sequences([clean(msg)]), maxlen=max_len, padding="post", truncating="post")
    p = float(model.predict(seq, verbose=0)[0][0])
    if p >= 0.5:
        st.error(f"SPAM  (probability {p:.2%})")
    else:
        st.success(f"HAM / legitimate  (spam probability {p:.2%})")
