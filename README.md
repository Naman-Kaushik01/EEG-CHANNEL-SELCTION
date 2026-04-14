# EEG Channel Selection using Machine Learning

This project demonstrates how to optimize EEG-based Brain-Computer Interfaces (BCI) by selecting the most important electrodes (channels) using Machine Learning.

## 🎯 Project Goal
Reduce the number of EEG channels (e.g., 32 → 8) while maintaining high classification accuracy. This reduces hardware cost and computational complexity.

## 📂 Project Structure
- `src/App.tsx`: The interactive web-based GUI (React).
- `eeg_project.py`: The full Python implementation for college submission.
- `metadata.json`: Project metadata.

## 🛠️ Features
1. **Data Simulation**: Generates synthetic EEG signals with embedded brain patterns.
2. **Preprocessing**: Bandpass filtering (4-45Hz) and Z-score normalization.
3. **Feature Extraction**: Calculates Mean, Variance, and Power for each channel.
4. **Channel Selection**: Implements Random Forest, RFE, and LASSO importance scoring.
5. **Evaluation**: Compares accuracy between "All Channels" and "Selected Channels".

## 🚀 How to Run the Python Code
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```
2. Run the script:
   ```bash
   python eeg_project.py
   ```

## 🧠 Concepts for Beginners
- **EEG**: Electrical signals from the brain.
- **Channels**: Different locations on the scalp where electrodes are placed.
- **Features**: Summary statistics (like average power) that describe the raw signal.
- **Channel Selection**: A form of "Feature Selection" where we pick the best sensors.
