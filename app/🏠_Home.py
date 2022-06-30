import streamlit as st

st.set_page_config(page_title="GSoC'22 @ TensorFlow",
                   page_icon="./app/images/tf_icon.png", layout='wide', initial_sidebar_state="expanded")

st.title("Google Summer of Code'22 @ TensorFlow")
header_cols = st.columns(3)
with header_cols[1]:
    header_cols[1].image('./app/images/iamger.png')

with st.expander(" 📝 Project Details", expanded=True):
    st.markdown("""
    
    ### Title: Develop Healthcare examples using TensorFlow
    This project is a part of Google Summer of Code 2022.
    
    [GSoC'22 @ TensorFlow Project Link](https://summerofcode.withgoogle.com/programs/2022/projects/2HAC6oqy)
    
    ### Project Mentor:
    - **[Josh Gordon](https://twitter.com/random_forests)**
    
    ## Objective
    
    Developing healthcare examples to showcase various use cases of deep learning algorithms and their applications. 
    These sample notebooks will provide students as well as the researchers an overview of the working of deep 
    learning algorithms in real-time scenarios. Further, these trained models will be primarily used on a 
    web-inference engine (currently under development) for underfunded medical sectors.""")

st.subheader("Inference Engines ⚙️:")


st.header('1. PANDA Segmentation')
st.caption('Description:')
st.write("""With more than 1 million new diagnoses reported every year, prostate cancer (PCa) is the second 
most common cancer among males worldwide that results in more than 350,000 deaths annually. The key to decreasing 
mortality is developing more precise diagnostics.""")
st.caption("Pseudo Segmentation Approach for Cancer Detection:")
st.info("""Pseudo Segmentation is a process of creation of fake mask maps by using the classification approach on the 
entire image at the patch level. The entire slide image is broken down into patches of fixed length and these patches 
are then classified. If found positive, that patch in the original image is then masked, thereby, creating a fake 
mask map.""")
with st.expander('Render Results'):
    st.image('./app/images/PandaSeg.png')


