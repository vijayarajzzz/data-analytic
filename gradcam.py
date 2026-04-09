import tensorflow as tf
import numpy as np
import cv2
import tensorflow as tf
import numpy as np

def generate_gradcam(model, img_array):

    model.trainable = True

    # Base MobileNetV2 model
    base_model = model.layers[0]

    # Last convolution layer inside MobileNetV2
    last_conv_layer = base_model.get_layer("Conv_1")

    # Create a new model ONLY from base_model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)

        # Now pass through remaining classifier layers manually
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x)

        class_index = tf.argmax(x[0])
        loss = x[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_heatmap(heatmap, original_image):

    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    return superimposed_img