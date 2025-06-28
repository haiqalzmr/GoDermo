# view_prediction_log.py
# -*- coding: utf-8 -*-
"""
This script reads and displays the prediction history
from ensemble_predictions_log.csv as a table.
"""

import pandas as pd
import os

log_path = 'logs/ensemble_predictions_log.csv'

if os.path.exists(log_path):
    df = pd.read_csv(log_path, header=None)
    df.columns = ['Timestamp', 'Image Path', 'Predicted Label', 'Confidence', 'Benign Prob', 'Malignant Prob']
    print("\n🧾 Ensemble Prediction Log:")
    print(df.tail(10).to_string(index=False))  # Show last 10 entries
else:
    print("\n⚠️ No log file found at 'logs/ensemble_predictions_log.csv'. Please run predictions first.")