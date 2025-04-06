# %% Imports
import torch
import torch.nn.functional as F
from interpret.blackbox import LimeTabular
from interpret import show

# %% lime
def lime_explain(X_train, X_test, y_test, model, num_instances=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def predict_fn(x):
        X_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = F.softmax(logits, dim = 1)
        return probs.cpu().numpy()

# %% flattening
    X_train_flat = X_train.view(X_train.shape[0], -1).numpy()
    X_test_flat = X_test.view(X_test.shape[0], -1).numpy()

# %% Apply lime
# Initilize Lime for Tabular data
    lime = LimeTabular(predict_fn=predict_fn, data=X_train_flat, random_state=42)
# Get local explanations
    lime_local = lime.explain_local(X_test_flat[-num_instances:], y_test[-num_instances:], name="LIME for LSTM")
    show(lime_local)

# %%
