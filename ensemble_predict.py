# ensemble_predict.py
import csv
import os
from datetime import datetime

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_labels = ['benign', 'malignant']

def load_models(use_mobilenet=True, use_efficientnet=True, use_densenet=True):
    models = []
    model_names = []
    if use_mobilenet:
        print("🔗 Using MobileNetV2")
        models.append(load_model('models/mobilenetv2_model.h5'))
        model_names.append("MobileNetV2")
    if use_efficientnet:
        print("🔗 Using EfficientNetV2B0")
        models.append(load_model('models/efficientnetv2b0_model.h5'))
        model_names.append("EfficientNetV2B0")
    if use_densenet:
        print("🔗 Using DenseNet121")
        models.append(load_model('models/densenet121_model.h5'))
        model_names.append("DenseNet121")
    return models, model_names

def ensemble_predict(img_path, use_mobilenet=True, use_efficientnet=True, use_densenet=True, weights=None):
    # Load individual models
    models, model_names = load_models(use_mobilenet, use_efficientnet, use_densenet)

    # Load ensemble model (final predictor)
    ensemble_model_path = "models/ensemble_model.h5"
    if not os.path.exists(ensemble_model_path):
        raise FileNotFoundError("Ensemble model not found. Make sure models/ensemble_model.h5 exists.")
    ensemble_model = load_model(ensemble_model_path)
    print("✅ Loaded Ensemble Model")

    # Prepare input
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Get predictions from individual models
    predictions = []
    outputs = {}

    for name, model in zip(model_names, models):
        pred = model.predict(img_array)[0]
        predictions.append(pred)
        print(f"{name} prediction: {pred}")
        outputs[name] = pred  # For PDF reporting

    # Use ensemble model prediction as final result
    ensemble_pred = ensemble_model.predict(img_array)[0]
    if isinstance(ensemble_pred, np.ndarray):
        ensemble_pred_scalar = float(ensemble_pred[0]) if isinstance(ensemble_pred, np.ndarray) else float(ensemble_pred)
    else:
        ensemble_pred_scalar = float(ensemble_pred)
    class_index = int(ensemble_pred_scalar > 0.5)
    confidence = ensemble_pred_scalar if class_index == 1 else 1 - ensemble_pred_scalar
    predicted_label = class_labels[class_index]
    outputs["EnsembleModel"] = [1 - ensemble_pred_scalar, ensemble_pred_scalar]  # For PDF: [benign, malignant]

    # Log prediction
    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/ensemble_predictions_log.csv'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Always log as a list
    if isinstance(ensemble_pred, np.ndarray):
        ensemble_pred_log = ensemble_pred.tolist()
    else:
        ensemble_pred_log = [ensemble_pred]
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, img_path, predicted_label, float(confidence)] + ensemble_pred_log)

    # Make sure all outputs are lists, not np.ndarray
    def sanitize_outputs(data):
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.generic,)):
            return data.item()
        elif isinstance(data, dict):
            return {k: sanitize_outputs(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_outputs(v) for v in data]
        else:
            return data

    outputs = sanitize_outputs(outputs)
    return predicted_label, float(confidence), ensemble_pred_scalar, outputs
