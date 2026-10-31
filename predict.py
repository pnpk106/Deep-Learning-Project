"""Classify a single SMS using the saved model.  Usage: python src/predict.py "your message" """
import sys, json, pickle, re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\b\d{5,}\b", " numtoken ", t)
    t = re.sub(r"[^a-z0-9\s£$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

model = tf.keras.models.load_model("models/spam_bilstm.keras")
tok = pickle.load(open("models/tokenizer.pkl", "rb"))
max_len = json.load(open("models/config.json"))["MAX_LEN"]
msg = " ".join(sys.argv[1:]) or input("Enter SMS: ")
seq = pad_sequences(tok.texts_to_sequences([clean(msg)]), maxlen=max_len, padding="post", truncating="post")
p = float(model.predict(seq, verbose=0)[0][0])
print(f"{'SPAM' if p >= 0.5 else 'HAM'}  (spam probability = {p:.4f})")
