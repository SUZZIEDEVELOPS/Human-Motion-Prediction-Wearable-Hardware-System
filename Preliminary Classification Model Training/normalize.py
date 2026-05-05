# src/normalize.py
#from sklearn.preprocessing import StandardScaler
#import joblib
#import os

#def normalize_features(X, save_path="models/scaler.pkl"):
 #   """
  #  Z-score standardization:
   # X_scaled = (X - mean) / std
    #"""
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    # Save scaler for later (e.g., testing/inference)
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #joblib.dump(scaler, save_path)

    #return X_scaled, scaler


#Split data → train / test
#Fit scaler only on training data
#Apply the same scaler to test data

from sklearn.preprocessing import StandardScaler
import joblib
import os

def normalize_features(X, scaler=None, save_path="models/scaler.pkl"):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


# from sklearn.preprocessing import StandardScaler
# import joblib
# import os
# import pandas as pd
#
# def normalize_features(X, scaler=None, save_path="models/scaler.pkl"):
#     if scaler is None:
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         joblib.dump(scaler, save_path)
#     else:
#         X_scaled = scaler.transform(X)
#
#     # Convert back to DataFrame to preserve feature names
#     X_scaled = pd.DataFrame(
#         X_scaled,
#         index=X.index,
#         columns=X.columns
#     )
#
#     return X_scaled, scaler
