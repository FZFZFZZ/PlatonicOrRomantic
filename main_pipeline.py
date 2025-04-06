import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
from preprocess import preprocess_data

#Step 1: preprocess
embeddings_file = "glove.6B.50d.txt"
output_pkl = "data.50d.pkl"
preprocess_data(embeddings_file, output_pkl, file="Data/train.csv")

# Step 2: load data
with open("data.50d.pkl", "rb") as f:
    data = pickle.load(f)

# Step 3: dialog -> vectors
def extract_features(data: list[tuple[list[tuple[int, np.ndarray | None]], int]]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for dialog, label in data:
        vectors = []
        for role, emb in dialog:
            if emb is not None:
                vectors.append(np.mean(emb, axis=0))
        if len(vectors) == 0:
            continue
        dialog_vector = np.mean(vectors, axis=0)
        X.append(dialog_vector)
        y.append(label)
    return np.array(X), np.array(y)

X, y = extract_features(data)

# Step 4: use dataloader!!
dl = DataLoader()
dl.X, dl.y = X, y
X_train, X_test, y_train, y_test = dl.get_data_split()
X_train, y_train = dl.oversample(X_train, y_train)

# Step 5: Train
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Step 6: Evaluate
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Step 7: LIME
lime = LimeTabular(predict_fn=rf.predict_proba, data=X_train, random_state=1)
lime_local = lime.explain_local(X_test[-20:], y_test[-20:], name='LIME')
show(lime_local)
