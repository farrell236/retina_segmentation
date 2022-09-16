import cv2

import numpy as np
import streamlit as st
import tensorflow as tf

from preprocessing.utils import _get_retina_bb, _pad_to_square


@st.cache(allow_output_mutation=True)
def load_model(model_file):
    model = tf.keras.models.load_model(model_file, compile=False)
    print(f'Model {model_file} Loaded!')
    return model


@st.cache(allow_output_mutation=True)
def load_gatekeeper():
    validator_model = tf.keras.models.load_model('EyeQ/ResNetV2-EyeQ-QA.tf')
    print('Gatekeeper Model Loaded!')
    return validator_model


def parse_function(image):
    image = tf.image.resize(image, [512, 512])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def main():

    st.title('Retina Segmentation')

    st.sidebar.title('Segmentation Model')
    options = st.sidebar.selectbox('Select Option:', ('Vessels', 'Lesions (BETA)'))
    gatekeeper = st.sidebar.radio("Gatekeeper:", ('Enabled', 'Disabled'))

    gatekeeper_model = load_gatekeeper()

    if options == 'Vessels':

        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader('Choose an image...', type=('png', 'jpg', 'jpeg'))

        model = load_model('checkpoints/DeeplabV3Plus_DRIVE.tf')

        if uploaded_file:
            col1, col2 = st.columns(2)

            # Load Image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Check image
            valid = np.argmax(gatekeeper_model(parse_function(image[None, ...])))
            if valid == 2 and gatekeeper == 'Enabled':
                st.image(image)
                st.info('Image is of poor quality')
                return

            # Localise and center retina image
            x, y, w, h, _ = _get_retina_bb(image)
            image = image[y:y + h, x:x + w, :]
            image = _pad_to_square(image, border=0)
            image = cv2.resize(image, (1024, 1024))

            with col1:
                st.subheader("Uploaded Image")
                st.image(image)

            # Apply CLAHE pre-processing
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Run model on input
            y_pred = model(image[None, ..., None])[0].numpy()

            with col2:
                st.subheader("Predicted Vessel")
                st.image(y_pred)

    elif options == 'Lesions (BETA)':

        st.write('```--- WARNING: This model is highly experimental ---```')

        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader('Choose an image...', type=('png', 'jpg', 'jpeg'))

        model = load_model('checkpoints/DeeplabV3Plus_FGADR.tf')

        if uploaded_file:
            col1, col2, col3, = st.columns(3)

            # Load Image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Check image
            valid = np.argmax(gatekeeper_model(parse_function(image[None, ...])))
            if valid == 2 and gatekeeper == 'Enabled':
                st.image(image)
                st.info('Image is of poor quality')
                return

            # Localise and center retina image
            x, y, w, h, _ = _get_retina_bb(image)
            image = image[y:y + h, x:x + w, :]
            image = _pad_to_square(image, border=0)
            image = cv2.resize(image, (1024, 1024))

            with col1:
                st.subheader("Uploaded Image")
                st.image(image)

            # Apply CLAHE pre-processing
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Run model on input
            y_pred = model(image[None, ..., None])[0].numpy()

            with col2:
                st.subheader(f'MA')
                st.image(y_pred[..., 1])
            with col3:
                st.subheader(f'HE')
                st.image(y_pred[..., 2])
            with col1:
                st.subheader(f'EX')
                st.image(y_pred[..., 3])
            with col2:
                st.subheader(f'SE')
                st.image(y_pred[..., 4])
            with col3:
                st.subheader(f'OD')
                st.image(y_pred[..., 5])

if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    main()
