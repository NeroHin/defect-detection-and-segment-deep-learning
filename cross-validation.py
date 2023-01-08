import numpy as np
from sklearn.model_selection import KFold


kfold = KFold(n_splits=4, shuffle=True, random_state=42)


X_train_folds = []
X_val_folds = []
y_train_folds = []
y_val_folds = []

# 將資料分為 4 個 fold
for train_index, val_index in kfold.split(X):
  X_train_folds.append(X[train_index])
  X_val_folds.append(X[val_index])
  y_train_folds.append(y[train_index])
  y_val_folds.append(y[val_index])


# Training and evaluating the model
for i in range(4):
  model = build_model() # Build the model
 
  
  print(f"Fold {i+1}, Score: {score}")
