import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.signal import butter, lfilter

# ==========================================
# 🎯 PROJECT: EEG Channel Selection using ML
# ==========================================

"""
EEG (Electroencephalogram) measures brain activity via electrodes (channels).
Channel selection helps reduce complexity (e.g., 32 channels -> 8) 
while keeping high classification accuracy.
"""

# ------------------------------------------
# 1. DATA SIMULATION (If real data not available)
# ------------------------------------------
def generate_simulated_eeg(n_channels=32, n_samples=1000, n_trials=100):
    """
    Creates a simulated EEG dataset.
    Structure: (Trials, Channels, Samples)
    """
    print(f"Generating simulated EEG data: {n_channels} channels, {n_trials} trials...")
    
    # Random noise + some "signal" in specific channels
    data = np.random.normal(0, 1, (n_trials, n_channels, n_samples))
    
    # Create two classes (e.g., Left vs Right hand movement)
    # Class 0: Stronger signal in channels 0-3
    # Class 1: Stronger signal in channels 10-13
    labels = np.zeros(n_trials)
    labels[n_trials//2:] = 1
    
    for i in range(n_trials):
        if labels[i] == 0:
            data[i, 0:4, :] += np.sin(np.linspace(0, 10, n_samples)) * 2
        else:
            data[i, 10:14, :] += np.sin(np.linspace(0, 10, n_samples)) * 2
            
    return data, labels

# ------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------
def bandpass_filter(data, lowcut=4.0, highcut=45.0, fs=250.0, order=5):
    """
    Removes noise outside the 4-45 Hz range (common for EEG).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='both')
    return lfilter(b, a, data)

def preprocess_eeg(data):
    """
    Applies filtering and normalization to each channel.
    """
    processed_data = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            # Filter
            filtered = bandpass_filter(data[trial, ch, :])
            # Normalize (Z-score)
            processed_data[trial, ch, :] = (filtered - np.mean(filtered)) / np.std(filtered)
    return processed_data

# ------------------------------------------
# 3. FEATURE EXTRACTION
# ------------------------------------------
def extract_features(data):
    """
    Converts raw time-series into simple numbers (features).
    - Mean: Average signal value
    - Variance: Signal spread (power)
    - Power: Sum of squares
    """
    n_trials, n_channels, _ = data.shape
    features = []
    
    for i in range(n_trials):
        trial_features = []
        for j in range(n_channels):
            ch_data = data[i, j, :]
            # Feature 1: Mean
            trial_features.append(np.mean(ch_data))
            # Feature 2: Variance
            trial_features.append(np.var(ch_data))
            # Feature 3: Standard Deviation
            trial_features.append(np.std(ch_data))
        features.append(trial_features)
        
    return np.array(features)

# ------------------------------------------
# 4. CHANNEL SELECTION METHODS
# ------------------------------------------
def select_channels(X, y, method='RF', n_top=8):
    """
    Identifies the most important EEG channels.
    """
    print(f"Selecting top {n_top} channels using {method}...")
    
    # Note: X has 3 features per channel, so we map them back to channels
    n_channels = X.shape[1] // 3
    
    if method == 'RF':
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        importances = model.feature_importances_
        # Average importance across the 3 features of each channel
        channel_importances = np.mean(importances.reshape(-1, 3), axis=1)
        
    elif method == 'RFE':
        estimator = LogisticRegression()
        selector = RFE(estimator, n_features_to_select=n_top * 3, step=1)
        selector = selector.fit(X, y)
        importances = selector.ranking_
        channel_importances = -np.mean(importances.reshape(-1, 3), axis=1) # Lower rank is better
        
    elif method == 'LASSO':
        model = Lasso(alpha=0.1)
        model.fit(X, y)
        importances = np.abs(model.coef_)
        channel_importances = np.mean(importances.reshape(-1, 3), axis=1)
        
    top_indices = np.argsort(channel_importances)[-n_top:]
    return top_indices, channel_importances

# ------------------------------------------
# 5. MAIN EXECUTION & EVALUATION
# ------------------------------------------
def run_project():
    # 1. Load/Generate Data
    raw_data, labels = generate_simulated_eeg(n_channels=32)
    
    # 2. Preprocess
    clean_data = preprocess_eeg(raw_data)
    
    # 3. Feature Extraction
    X = extract_features(clean_data)
    y = labels
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Baseline (All Channels)
    clf_all = RandomForestClassifier(random_state=42)
    clf_all.fit(X_train, y_train)
    acc_all = accuracy_score(y_test, clf_all.predict(X_test))
    print(f"\n✅ Accuracy with ALL 32 channels: {acc_all*100:.2f}%")
    
    # 6. Channel Selection
    top_channels, importances = select_channels(X, y, method='RF', n_top=8)
    print(f"Selected Channels: {top_channels}")
    
    # 7. Train with Selected Channels
    # Map channel indices back to feature indices
    feature_indices = []
    for ch in top_channels:
        feature_indices.extend([ch*3, ch*3+1, ch*3+2])
        
    X_train_sel = X_train[:, feature_indices]
    X_test_sel = X_test[:, feature_indices]
    
    clf_sel = RandomForestClassifier(random_state=42)
    clf_sel.fit(X_train_sel, y_train)
    acc_sel = accuracy_score(y_test, clf_sel.predict(X_test_sel))
    print(f"✅ Accuracy with TOP 8 channels: {acc_sel*100:.2f}%")
    
    # 8. Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Feature Importance
    plt.subplot(1, 2, 1)
    plt.bar(range(32), importances)
    plt.title("Channel Importance Score")
    plt.xlabel("Channel Index")
    plt.ylabel("Score")
    
    # Plot 2: Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.bar(['All Channels', 'Top 8 Channels'], [acc_all, acc_sel], color=['blue', 'green'])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_project()
