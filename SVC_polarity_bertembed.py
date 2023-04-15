import os
import random
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

stopwords = set(stopwords.words("english"))


from transformers import AutoTokenizer, AutoModel
import torch

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_data_from_directory(data_dir):
    data = []
    for label, sentiment in enumerate(["pos", "neg"]):
        sentiment_dir = os.path.join(data_dir, sentiment)
        for filename in os.listdir(sentiment_dir):
            with open(os.path.join(sentiment_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                data.append({"text": text, "label": label})
    return pd.DataFrame(data)

data_dir = "/storage/users/ljcc0930/workspace/CSE-842-hw/data/txt_sentoken"

df = load_data_from_directory(data_dir)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

def sentence_vector_bert(sentence, model, tokenizer, stopwords=None, device="cpu"):
    if stopwords:
        words = word_tokenize(sentence)
        words = [word for word in words if word not in stopwords]
        sentence = " ".join(words)

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings.squeeze()

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model.to(device)

X_train = np.vstack(train_df["text"].apply(lambda x: sentence_vector_bert(x, bert_model, bert_tokenizer, stopwords, device)))
X_val = np.vstack(val_df["text"].apply(lambda x: sentence_vector_bert(x, bert_model, bert_tokenizer, stopwords, device)))

y_train = train_df["label"]
y_val = val_df["label"]

print("SVC Starting!!")
svc = SVC(random_state=SEED)

param_grid = {"C": [3, 5, 7, 0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ('linear', 'rbf')}
grid_search = GridSearchCV(svc, param_grid, cv=3, scoring="accuracy", n_jobs=20)
grid_search.fit(X_train, y_train)

print(f"Embedding: BERT")
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

best_svc = grid_search.best_estimator_
y_val_pred = best_svc.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation accuracy: {val_accuracy:.4f}\n")

report = classification_report(y_val, y_val_pred, target_names=["pos", "neg"])
print("\nClassification Report:")
print(report)

