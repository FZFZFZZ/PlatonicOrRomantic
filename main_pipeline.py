import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Step 1: Data Preprocessing
def preprocess_data_pipeline(embeddings_file, output_pkl, train_file):
    preprocess_data(embeddings_file, output_pkl, file=train_file)
    with open(output_pkl, "rb") as f:
        data = pickle.load(f)
    return data

# Step 2: Extract Features from Data
def extract_features(data):
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

# Step 3: Split Data and Oversample (if needed)
def prepare_data(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample the training set if needed (to handle class imbalance)
    oversampler = RandomOverSampler(random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

# Step 4: Train a RandomForest model
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

# Step 5: Evaluate the Model
def evaluate_model(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    return f1, accuracy

# Step 6: Explain the model with LIME
def lime_explanation(rf, X_train, X_test, y_test, num_instances=20):
    lime = LimeTabular(predict_fn=rf.predict_proba, data=X_train, random_state=1)
    lime_local = lime.explain_local(X_test[-num_instances:], y_test[-num_instances:], name='LIME Explanation')
    show(lime_local)

# Main Pipeline Function
def main_pipeline(embeddings_file, train_file, output_pkl, num_instances=20):
    # Step 1: Preprocess the Data
    data = preprocess_data_pipeline(embeddings_file, output_pkl, train_file)

    # Step 2: Extract Features
    X, y = extract_features(data)

    # Step 3: Prepare Data (train-test split, oversampling)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Step 4: Train RandomForest Classifier
    rf = train_random_forest(X_train, y_train)

    # Step 5: Evaluate the Model
    f1, accuracy = evaluate_model(rf, X_test, y_test)

    # Step 6: Explain the Model with LIME
    lime_explanation(rf, X_train, X_test, y_test, num_instances)

# Main entry point for the script
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python mainpipeline.py <embeddings file> <train file> <output pkl file> <num_instances>")
        sys.exit(1)

    embeddings_file = sys.argv[1]
    train_file = sys.argv[2]
    output_pkl = sys.argv[3]
    num_instances = int(sys.argv[4])

    main_pipeline(embeddings_file, train_file, output_pkl, num_instances)
