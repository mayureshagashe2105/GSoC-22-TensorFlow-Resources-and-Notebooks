# GSoC'22 @ TensorFlow

<p align="center">
<img src="assets/images/iamger.png" alt="image"/>
</p>


## Project Details:

### Develop Healthcare examples using TensorFlow
This project is a part of Google Summer of Code 2022.

[GSoC'22 @ TensorFlow Project Link](https://summerofcode.withgoogle.com/programs/2022/projects/2HAC6oqy)


#### Work-Product Document (Final Report):
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mayureshagashe2105/gsoc22-tensorflow-wrapping-up-484d326e7235)


### Project Mentor:
- **[Josh Gordon](https://twitter.com/random_forests)**

## Objective

Developing healthcare examples to showcase various use cases of deep learning algorithms and their applications. These sample notebooks will provide students as well as the researchers an overview of the working of deep learning algorithms in real-time scenarios. Further, these trained models will be primarily used on a web-inference engine (currently under development) for underfunded medical sectors.


## Pseudo-segmentation of Prostate Gland Cancer
### Understanding the Problem Statement
Pseudo Segmentation is a process of creation of fake mask maps by using the classification approach on the entire image at the patch level. The entire slide image is broken down into patches of fixed length and these patches are then classified. If found positive, that patch in the original image is then masked, thereby, creating a fake mask map.


### Demo:

![Untitled](https://user-images.githubusercontent.com/75118658/179668154-983e9d95-a324-4a1d-af9b-e67f17b97e15.gif)

## Access the deployed App

App:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mayureshagashe2105-gsoc-22-tensorflow-resources-app-home-esfrdc.streamlitapp.com/)

## Run the App Locally
1. Clone the repository.
```sh
git clone https://github.com/mayureshagashe2105/GSoC-22-TensorFlow-Resources-and-Notebooks.git
```
2. Go to the project directory.
```sh
cd GSoC-22-TensorFlow-Resources-and-Notebooks
```
4. Checkout to the branch `localhost`
```sh
git checkout localhost
```
5. Go to the `app` driectory.
```sh
cd app
```
6. Install the requiremnets.
```sh
pip install -r requirements.txt
```
7. Make sure you have `openslide-binary` from this [link](http://openslide.org/download/#windows-binaries)
8. Run the following command:
```sh
streamlit run 🏠_Home.py
```



## This blog post presents the technical insights used in the developed diagnosing method.
Blog Post:


[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mayureshagashe2105/gsoc22-tensorflow-pseudo-segmentation-of-prostate-gland-tissue-for-cancer-detection-6d5c45c7c46a)

## Timeline
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
- [x] Optimizing patch extraction process.
- [x] Try to simplify the codebase.
- [x] Document the approach used.

### Week 5:
#### Tasks
- [x] MLPs with JAX (Batch_mode)
- [x] CNNs with JAX
- [x] ViTs with JAX-Flax.

### Week 6:
#### Tasks
- [x] Figure out how to use ViTs with patch based learning.
- [x] Fix bug in score function from ViT.
- [x] Use optax for optimizer state handling.
- [x] Fix inappropriate accuracy bug.
- [x] Document the ViTs.

### Week 7 - 9:
#### Tasks:
- [x] Add docstrings to ViTs.
- [x] Add dropout layer and support for dropout_rng.
- [x] Add tensorboard pulgin.

### Week 10 - 11:
#### Tasks:
- [x] Publish a release.


### Week 12:
#### Tasks:
- [x] Final Work product document write-up.

