# app.py
# -*- coding: utf-8 -*-
"""
This is the final Flask app for GoDermo demo.
Features:
- Image upload
- Ensemble prediction
- Grad-CAM heatmap
- PDF report generation
- Result page with download
"""

import io
import os
import uuid
from datetime import datetime

import numpy as np
from flask import (Flask, flash, redirect, render_template, request, send_file,
                   session, url_for)
from flask_migrate import Migrate
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Warning: TensorFlow not installed. Some features may not work.")
from werkzeug.utils import secure_filename

from auth import auth
from ensemble_predict import ensemble_predict
from generate_report import generate_pdf_report
from gradcam import generate_gradcam
from models import Result, User, db

app = Flask(__name__)
app.register_blueprint(auth)
app.secret_key = 'your-secret-key'

UPLOAD_FOLDER = 'static/uploaded_images'
HEATMAP_FOLDER = 'static/heatmaps'
REPORT_FOLDER = 'static/reports'

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'instance', 'GoDermo.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "Healthcheck passed!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No image selected", 400

    if file.filename is None:
        return "Invalid filename", 400
        
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(image_path)

    # Run prediction
    label, confidence, ensemble_pred, model_outputs = ensemble_predict(
        image_path,
        use_mobilenet=True,
        use_efficientnet=True,
        use_densenet=True
    )

    # Generate readable report filename AFTER label is known
    readable_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"GoDermo_Report_{label.upper()}_{readable_time}.pdf"
    report_path = os.path.join(app.config['REPORT_FOLDER'], base_name)

    # Grad-CAM generation using best model from ensemble
    model_names = ["mobilenetv2", "efficientnetv2b0", "densenet121"]
    model_paths = [
        "models/mobilenetv2_model.h5",
        "models/efficientnetv2b0_model.h5",
        "models/densenet121_model.h5"
    ]
    last_conv_layers = {
        "mobilenetv2": "Conv_1",
        "efficientnetv2b0": "top_activation",
        "densenet121": "conv5_block16_concat"
    }
    # Find the model that had highest individual confidence
  
    # Grad-CAM will use MobileNetV2 branch of ensemble model
    last_conv_layer_name = "block_13_expand"
    model = load_model("models/ensemble_model.h5")

    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], f"heatmap_{unique_filename}")
    class_index = 0 if label.lower() == 'benign' else 1

    # Call updated Grad-CAM
    generate_gradcam(
        model=model,
        img_path=image_path,
        save_path=heatmap_path,
        class_index=class_index,
        last_conv_layer_name=last_conv_layer_name
    )


    # PDF Report
    generate_pdf_report(
        image_path=image_path,
        heatmap_path=heatmap_path,
        prediction=label,
        confidence=confidence * 100,
        output_path=report_path,
        model_outputs=model_outputs
    )
    
    # Save prediction to database
    if 'user_id' in session:
        user_id = session.get('user_id')
        if not user_id:
            flash('You must be logged in to submit a prediction.', 'danger')
            return redirect(url_for('auth.login'))
        
        result = Result(
            image_path=image_path,
            result=label,
            confidence=confidence,
            timestamp=datetime.now(),
            user_id=user_id,
            heatmap_path=heatmap_path,
            mobilenet_prob=format_probs(model_outputs.get("MobileNetV2")),
            efficientnet_prob=format_probs(model_outputs.get("EfficientNetV2B0")),
            densenet_prob=format_probs(model_outputs.get("DenseNet121")),
            ensemble_prob=format_probs(model_outputs.get("EnsembleModel")),
            report_path=report_path
        )

        db.session.add(result)
        db.session.commit()
    
    return render_template('result.html',
        result=label + " (Ensemble Model)",
        confidence=round(confidence * 100, 2),
        image_path='/' + image_path.replace('\\', '/'),
        heatmap_path='/' + heatmap_path.replace('\\', '/'),
        report_path='/' + report_path.replace('\\', '/')
    )

@app.route('/history')
def history():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('auth.login'))
    results = Result.query.filter_by(user_id=user_id).order_by(Result.timestamp.desc()).all()
    predictions = []
    for r in results:
        # If you have report_path, use it; else set to None or build it as needed
        predictions.append((
            r.id,
            r.image_path,
            r.result,
            r.confidence,
            r.report_path,
            r.timestamp.strftime('%Y-%m-%d %H:%M:%S') if r.timestamp else ''
        ))
    return render_template('history.html', predictions=predictions)

@app.route('/delete/<int:prediction_id>', methods=['POST'])
def delete_prediction_route(prediction_id):
    image_path = request.form.get('image_path')
    
    
    # Delete from Result table
    result = Result.query.get(prediction_id)
    if result:
        db.session.delete(result)
        db.session.commit()
    
    # Optionally delete the image file
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    
    flash("Prediction deleted successfully.", "success")
    return redirect(url_for('history'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    user = User.query.get(user_id)
    if request.method == 'POST':
        user.first_name = request.form['first_name']
        user.last_name = request.form['last_name']
        user.email = request.form['email']
        user.age = request.form['age']
        user.gender = request.form['gender']
        if request.form['password']:
            user.password = generate_password_hash(request.form['password'])
        db.session.commit()
        # Update session variable for navbar greeting
        session['user_name'] = user.first_name
        flash('Profile updated successfully.')
        return redirect(url_for('profile'))
    return render_template('profile.html', user=user)

@app.route('/generate_report/<int:result_id>')
def generate_report(result_id):
    import ast

    result = Result.query.get(result_id)
    if not result:
        return "Result not found", 404

    # Prepare paths
    image_path = result.image_path
    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], f"heatmap_{os.path.basename(image_path)}")
    if not os.path.exists(heatmap_path):
        heatmap_path = None

    # Prepare filename
    result_label = result.result.upper()
    timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"GoDermo_Report_{result_label}_{timestamp}.pdf"

    # Helper to parse stored stringified lists (if needed)
    def parse_probs(probs):
        if probs is None:
            return None
        if isinstance(probs, str) and ',' in probs:
            try:
                return [float(p.strip()) for p in probs.split(',')]
            except:
                return None
        return None


    # Reconstruct model_outputs dictionary
    model_outputs = {
        "MobileNetV2": parse_probs(getattr(result, 'mobilenet_prob', None)),
        "EfficientNetV2B0": parse_probs(getattr(result, 'efficientnet_prob', None)),
        "DenseNet121": parse_probs(getattr(result, 'densenet_prob', None)),
        "EnsembleModel": parse_probs(getattr(result, 'ensemble_prob', None))
    }

    # Only include models with valid probabilities
    formatted_model_outputs = {}
    for model_name, probs in model_outputs.items():
        if probs and len(probs) == 2:
            formatted_model_outputs[model_name] = probs
        elif probs and len(probs) == 1:
            # If only one value, assume it's malignant and benign is 1-x
            formatted_model_outputs[model_name] = [1 - float(probs[0]), float(probs[0])]

    # Generate the PDF in memory
    from generate_report import generate_pdf_report
    pdf_buffer = io.BytesIO()
    generate_pdf_report(
        image_path=image_path,
        heatmap_path=heatmap_path,
        prediction=result.result,
        confidence=float(result.confidence) * 100,
        output_path=pdf_buffer,
        model_outputs=formatted_model_outputs
    )
    pdf_buffer.seek(0)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

def format_probs(prob):
    try:
        return f"{float(prob[0]):.6f},{float(prob[1]):.6f}"
    except:
        return "0.000000,0.000000"

if __name__ == '__main__':
    app.run(debug=True)
