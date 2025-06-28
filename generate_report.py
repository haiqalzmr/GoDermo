import os
from datetime import datetime

from fpdf import FPDF


def generate_pdf_report(image_path, heatmap_path, prediction, confidence, output_path, model_outputs=None):
    pdf = FPDF()
    pdf.add_page()

    # Title and summary
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Skin Lesion Detection Report", ln=True, align='C')
    pdf.ln(8)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Final Prediction: {prediction.upper()}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(8)

    # Images (smaller side-by-side)
    image_width = 75
    spacing = 20
    x_start = 15
    y_start = pdf.get_y()

    if os.path.exists(image_path):
        pdf.set_xy(x_start, y_start)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(image_width, 8, "Original Image", ln=2)
        pdf.image(image_path, x=x_start, w=image_width)

    if os.path.exists(heatmap_path):
        x2 = x_start + image_width + spacing
        pdf.set_xy(x2, y_start)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(image_width, 8, "Grad-CAM Heatmap", ln=2)
        pdf.image(heatmap_path, x=x2, w=image_width)

    # Move down below images
    pdf.set_y(y_start + image_width + 10)

    if model_outputs:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Model Output Probabilities", ln=True)
        pdf.set_font("Arial", '', 11)

        for model, probs in model_outputs.items():
            if isinstance(probs, str) and ',' in probs:
                probs = [float(p.strip()) for p in probs.split(',')]
            if isinstance(probs, list) and len(probs) == 2:
                pdf.ln(2)
                pdf.cell(0, 8, f"{model}:", ln=True)
                pdf.cell(0, 8, f"   Benign: {probs[0]*100:.2f}%", ln=True)
                pdf.cell(0, 8, f"Malignant: {probs[1]*100:.2f}%", ln=True)


    pdf.output(output_path)
    print(f"✅ PDF report saved to: {output_path}")
