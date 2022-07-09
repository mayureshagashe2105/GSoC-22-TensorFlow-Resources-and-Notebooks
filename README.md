# GSoC'22 @ TensorFlow

<p align="center">
<img src="assets/images/iamger.png" alt="image"/>
</p>


## Project Details:

### Develop Healthcare examples using TensorFlow
This project is a part of Google Summer of Code 2022.

[GSoC'22 @ TensorFlow Project Link](https://summerofcode.withgoogle.com/programs/2022/projects/2HAC6oqy)

### Project Mentor:
- **[Josh Gordon](https://twitter.com/random_forests)**

## Objective

Developing healthcare examples to showcase various use cases of deep learning algorithms and their applications. These sample notebooks will provide students as well as the researchers an overview of the working of deep learning algorithms in real-time scenarios. Further, these trained models will be primarily used on a web-inference engine (currently under development) for underfunded medical sectors.

## Timeline

### Pseudo-segmentation of Prostate Gland Cancer
App:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mayureshagashe2105-gsoc-22-tensorflow-resources-app-home-nrud12.streamlitapp.com/)

Blog Post:
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mayureshagashe2105/gsoc22-tensorflow-pseudo-segmentation-of-prostate-gland-tissue-for-cancer-detection-6d5c45c7c46a)

### Week 1 - 2:
#### Tasks
- [x] Understanding the structure of the data and tiff files.
- [x] Dataset is hosted on kaggle and is very huge (412 GB). In order to get started, write a script to download the subset of the data from the kaggle environment.
- [x] Perform basic EDA.
- [x] Design custom data generators to ingest the data at the patch-level and benchmark the data generators.
- [x] Train a baseline model.
- [x] Create fake segmentation/color maps on the basis of classification results from the baseline model.
- [x] Optimize the Datagenerators for level 0 patch extraction.
- [x] Add `write to disk` functionality to Datagenerators.
- [x] Map classification results at higer resolution to segmentation map at a lower resolution.

### Week 3-4:
#### Tasks
- [x] Benchmarking the Input pipeline.
- [x] Depicting Diagramatic Explanantions.
- [ ] Optimizing patch extraction process.
- [x] Try to simplify the codebase.
- [x] Document the approach used.
