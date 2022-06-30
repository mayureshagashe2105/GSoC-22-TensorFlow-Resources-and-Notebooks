import streamlit as st
import openslide

from services.PANDASegmentation import *

st.set_page_config(page_title="PANDA Segmentation",
                   page_icon="./app/images/tf_icon.png", layout='wide', initial_sidebar_state="expanded")

st.title('Prostate Gland Cancer Segmentation Engine')
st.sidebar.info('Set the parameters and upload the WSI to get the inference.')

uploaded_WSI = st.file_uploader("Upload the Whole Slide Image (WSI) of the Prostate tissue", ['tiff', 'tif'])

if uploaded_WSI is not None:
    tiff_encoded_image = uploaded_WSI.read()
    with open('./app/images/test_sub.tiff', 'wb') as f:
        f.write(tiff_encoded_image)
    test = openslide.OpenSlide('./app/images/test_sub.tiff')

    st.sidebar.markdown("## Params for Inference")
    index_level = st.sidebar.selectbox('Index_level', [0, 1, 2])
    seg_res = st.sidebar.checkbox('Show Segmented Results', True)

    cols = st.columns(2)

    with cols[0]:
        x = int(cols[0].number_input('Width of Patch Size', 128, test.level_dimensions[index_level][0], step=1))

    with cols[1]:
        y = int(cols[1].number_input('Height of Patch Size', 128, test.level_dimensions[index_level][1], step=1))

    cols1 = st.columns(2)

    with cols1[0]:
        tissue_thresh = cols1[0].slider('Tissue Threshold %', 0.0, 1.0, step=0.01, value=0.35)

    with cols[1]:
        confidence_level = cols1[1].slider('Confidence_level %', 0.5, 1.0, step=0.01, value=0.75)

    sub = st.button('Get Inference')

    if sub:
        PANDASegmentation.get_inference(test, index_level, (x, y), confidence_level, tissue_thresh, seg_res)
