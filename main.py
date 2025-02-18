import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import io

def load_image(image_file):
    image = PIL.Image.open(image_file)
    image = image.resize((512, 512))  # Resize for performance
    image = np.array(image)  # No normalization yet
    if image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
        image = image[..., :3]
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.0  # Normalize after conversion
    return tf.expand_dims(image, axis=0)  # Add batch dimension

def tensor_to_image(tensor):
    tensor = tensor.numpy()  # Convert to NumPy array
    tensor = np.squeeze(tensor, axis=0)  # Remove batch dimension
    tensor = (tensor * 255).astype(np.uint8)  # Scale and convert to uint8
    return PIL.Image.fromarray(tensor)

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    matrix = tf.reshape(tensor, [-1, channels])
    return tf.matmul(matrix, matrix, transpose_a=True) / tf.cast(tf.shape(matrix)[0], tf.float32)

def style_transfer(content, style, style_strength):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    layers = content_layers + style_layers
    outputs = [vgg.get_layer(name).output for name in layers]
    model = tf.keras.Model([vgg.input], outputs)
    
    content_features = model(content)
    style_features = model(style)
    
    content_target = content_features[0]
    style_grams = [gram_matrix(style_layer) for style_layer in style_features[1:]]
    
    stylized_image = tf.Variable(content)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            outputs = model(stylized_image)
            content_loss = tf.reduce_mean((outputs[0] - content_target) ** 2)
            style_loss = sum(tf.reduce_mean((gram_matrix(output) - gram) ** 2)
                             for output, gram in zip(outputs[1:], style_grams))
            
            color_loss = tf.reduce_mean((stylized_image - style) ** 2)  # Ensure color influence
            loss = content_loss + (style_strength * 10.0) * style_loss + (style_strength * 5.0) * color_loss
        
        grad = tape.gradient(loss, stylized_image)
        optimizer.apply_gradients([(grad, stylized_image)])
        stylized_image.assign(tf.clip_by_value(stylized_image, 0.0, 1.0))
    
    for _ in range(50):
        train_step()
    
    return tensor_to_image(stylized_image)

st.set_page_config(page_title="Style Transfer App", layout="wide")
st.title("Neural Style Transfer")

col1, col2 = st.columns(2)
with col1:
    content_image = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
    if content_image:
        content = load_image(content_image)
        st.image(content_image, caption="Content Image", use_column_width=True)

with col2:
    style_image = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
    if style_image:
        style = load_image(style_image)
        st.image(style_image, caption="Style Image", use_column_width=True)

style_strength = st.slider("Style Intensity", 0.0, 1.0, 0.5)

if content_image and style_image:
    if st.button("Apply Style Transfer"):
        with st.spinner("Processing..."):
            output_image = style_transfer(content, style, style_strength)
            st.image(output_image, caption="Stylized Image", use_column_width=True)
            
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            st.download_button(label="Download Image", data=buf.getvalue(), file_name="stylized.png", mime="image/png")
