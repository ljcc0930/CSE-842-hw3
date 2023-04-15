# import os
# import random
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score


# vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
# X_train = vectorizer.fit_transform(train_df["text"])
# X_val = vectorizer.transform(val_df["text"])

# y_train = train_df["label"]
# y_val = val_df["label"]

# svc = SVC(random_state=SEED)

# param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
# grid_search = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print("Best parameters: ", grid_search.best_params_)
# print("Best cross-validation score: ", grid_search.best_score_)

# best_svc = grid_search.best_estimator_
# y_val_pred = best_svc.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)

# print(f"Validation accuracy: {val_accuracy:.4f}")

import os
import random
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from gensim.models import KeyedVectors
from allennlp.modules.elmo import Elmo, batch_to_ids

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

def load_data_from_directory(data_dir):
    data = []
    for label, sentiment in enumerate(["pos", "neg"]):
        sentiment_dir = os.path.join(data_dir, sentiment)
        for filename in os.listdir(sentiment_dir):
            with open(os.path.join(sentiment_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                data.append({"text": text, "label": label})
    return pd.DataFrame(data)

def load_glove_model(file):
    model = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            model[word] = vector
    return model

data_dir = "/storage/users/ljcc0930/workspace/CSE-842-hw/data/txt_sentoken"

df = load_data_from_directory(data_dir)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

nltk.download("punkt")
nltk.download("stopwords")

# Load and preprocess the dataset as before

def word2vec_sentence_vector(sentence, model, stopwords=None):
    words = word_tokenize(sentence)
    if stopwords:
        words = [word for word in words if word not in stopwords]
    vector = np.mean([model[word] for word in words if word in model.vocab], axis=0)
    return vector

def elmo_sentence_vector(sentence, model, stopwords=None):
    words = word_tokenize(sentence)
    if stopwords:
        words = [word for word in words if word not in stopwords]
    character_ids = batch_to_ids([words]).cuda()
    embeddings = model(character_ids)["elmo_representations"][0].mean(dim=1).detach().cpu().numpy()
    return embeddings.squeeze()

def glove_sentence_vector(sentence, model, stopwords=None):
    words = word_tokenize(sentence)
    if stopwords:
        words = [word for word in words if word not in stopwords]
    vector = np.mean([model[word] for word in words if word in model], axis=0)
    return vector

stopwords = set(stopwords.words("english"))

# Load Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

# Load GloVe model
# glove_model = load_glove_model("glove.6B.50d.txt")

# Load ELMo model
# options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False).cuda()

embedding_methods = {
    "Word2Vec": word2vec_model,
    "GloVe": glove_model,
    # "ELMo": elmo
}

for name, model in embedding_methods.items():
    if name == "ELMo":
        X_train = np.vstack(train_df["text"].apply(lambda x: elmo_sentence_vector(x, model, stopwords)))
        X_val = np.vstack(val_df["text"].apply(lambda x: elmo_sentence_vector(x, model, stopwords)))
    elif name == "Word2Vec":
        X_train = np.vstack(train_df["text"].apply(lambda x: word2vec_sentence_vector(x, model, stopwords)))
        X_val = np.vstack(val_df["text"].apply(lambda x: word2vec_sentence_vector(x, model, stopwords)))
    elif name == "GloVe":
        X_train = np.vstack(train_df["text"].apply(lambda x: glove_sentence_vector(x, model, stopwords)))
        X_val = np.vstack(val_df["text"].apply(lambda x: glove_sentence_vector(x, model, stopwords)))

    y_train = train_df["label"]
    y_val = val_df["label"]

    print("SVC Starting!!")
    svc = SVC(random_state=SEED)

    param_grid = {"C": [3, 5, 7, 0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ('linear', 'rbf')}
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring="accuracy", n_jobs=20)
    grid_search.fit(X_train, y_train)

    print(f"Embedding: {name}")
    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    best_svc = grid_search.best_estimator_
    y_val_pred = best_svc.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Validation accuracy: {val_accuracy:.4f}\n")

