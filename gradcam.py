import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

def generate_gradcam(model, img_path, save_path, class_index, last_conv_layer_name=None, colormap=None, blend=0.5):
    # Assign default colormap AFTER cv2 is imported
    if colormap is None:
        colormap = cv2.COLORMAP_JET

    # Auto-assign default last conv layer name
    if last_conv_layer_name is None:
        if "mobilenetv2" in model.name.lower():
            last_conv_layer_name = "Conv_1"
        elif "efficientnet" in model.name.lower():
            last_conv_layer_name = "top_conv"
        elif "densenet" in model.name.lower():
            last_conv_layer_name = "conv5_block16_concat"
        else:
            raise ValueError("❌ Please specify last_conv_layer_name explicitly for this model.")

    # Step 1: Preprocess input
    img_input = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_input)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Step 2: Original image for overlay
    original_img = Image.open(img_path).convert('RGB')
    original_img_np = np.array(original_img)
    height, width, _ = original_img_np.shape

    # Step 3: Grad-CAM model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Step 4: Gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_output = predictions[:, 0]


    grads = tape.gradient(class_output, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = np.power(heatmap, 0.5)  # Boost contrast


    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Step 5: Blend overlay
    superimposed_img = cv2.addWeighted(original_img_np, 1 - blend, heatmap, blend, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, superimposed_img)
    print(f"✅ Grad-CAM saved at: {save_path}")
    
def make_gradcam_heatmap_from_ensemble(img_array, model, conv_layer_name='block_13_expand'):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()