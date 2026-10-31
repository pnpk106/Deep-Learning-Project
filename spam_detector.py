"""
SMS Spam Detection using a Bidirectional LSTM (BiLSTM)
Dataset: SMS Spam Collection (UCI Machine Learning Repository, ID 228)
Run:   python src/spam_detector.py
Optional: place the raw file at data/SMSSpamCollection (tab separated: label<TAB>text)
          if you are offline; the script will use it automatically.
"""
import os, re, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
MAX_WORDS = 10000      # vocabulary size
MAX_LEN = 100          # tokens per message after padding/truncation
EMBED_DIM = 64
EPOCHS = 20
BATCH_SIZE = 32
OUT_MODEL, OUT_RES = "models", "results"
os.makedirs(OUT_MODEL, exist_ok=True); os.makedirs(OUT_RES, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------------------------------------------------------- 1. LOAD DATA
def load_dataset():
    local = os.path.join("data", "SMSSpamCollection")
    if os.path.exists(local):
        df = pd.read_csv(local, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
        print("Loaded local file:", local)
        return df
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=228)                     # SMS Spam Collection
    X, y = ds.data.features, ds.data.targets
    df = pd.concat([X, y], axis=1)
    text_col = X.columns[0]
    label_col = y.columns[0]
    df = df[[label_col, text_col]]; df.columns = ["label", "text"]
    print("Loaded from UCI repository (ID 228)")
    return df

df = load_dataset()
df = df.dropna().drop_duplicates().reset_index(drop=True)
df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"]); df["label"] = df["label"].astype(int)
print("Records after cleaning:", len(df)); print(df["label"].value_counts().rename({0: "ham", 1: "spam"}))

# ---------------------------------------------------------------- 2. PREPROCESS
def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\b\d{5,}\b", " numtoken ", t)          # long numbers / phone numbers
    t = re.sub(r"[^a-z0-9\s£$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

df["clean"] = df["text"].apply(clean)

X_train, X_tmp, y_train, y_tmp = train_test_split(df["clean"], df["label"], test_size=0.30,
                                                  stratify=df["label"], random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                stratify=y_tmp, random_state=SEED)
print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tok.fit_on_texts(X_train)                                  # fit ONLY on training data
def encode(s): return pad_sequences(tok.texts_to_sequences(s), maxlen=MAX_LEN, padding="post", truncating="post")
Xtr, Xva, Xte = encode(X_train), encode(X_val), encode(X_test)
ytr, yva, yte = y_train.values, y_val.values, y_test.values

# ---------------------------------------------------------------- 3. MODEL
model = models.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(MAX_WORDS, EMBED_DIM),
    layers.SpatialDropout1D(0.3),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
model.summary()

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
class_weight = {0: cw[0], 1: cw[1]}                         # handles ham/spam imbalance
cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
       callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]

# ---------------------------------------------------------------- 4. TRAIN
hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH_SIZE,
                 class_weight=class_weight, callbacks=cbs, verbose=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT_RES}/training_curves.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. EVALUATE
prob = model.predict(Xte, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
metrics = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
           "recall": recall_score(yte, pred), "f1_score": f1_score(yte, pred),
           "roc_auc": roc_auc_score(yte, prob)}
print("\nTEST METRICS")
for k, v in metrics.items():
    print(f"{k:10s}: {v:.4f}")
print("\n", classification_report(yte, pred, target_names=["ham", "spam"]))
json.dump(metrics, open(f"{OUT_RES}/metrics.json", "w"), indent=2)

cm = confusion_matrix(yte, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT_RES}/confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(yte, prob)
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_RES}/roc_curve.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. SAVE
model.save(f"{OUT_MODEL}/spam_bilstm.keras")
pickle.dump(tok, open(f"{OUT_MODEL}/tokenizer.pkl", "wb"))
json.dump({"MAX_WORDS": MAX_WORDS, "MAX_LEN": MAX_LEN}, open(f"{OUT_MODEL}/config.json", "w"))

# ---------------------------------------------------------------- 7. PREDICT
def predict_sms(message, threshold=0.5):
    p = float(model.predict(encode([clean(message)]), verbose=0)[0][0])
    return {"message": message, "label": "SPAM" if p >= threshold else "HAM", "spam_probability": round(p, 4)}

if __name__ == "__main__":
    for m in ["Congratulations! You have won a free prize. Claim now at www.win-cash.com",
              "Hey, are we still meeting for lunch at 1pm tomorrow?"]:
        print(predict_sms(m))
