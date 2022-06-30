import tensorflow as tf
import streamlit as st
import numpy as np


class PANDASegmentation:

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def __get_model(model_path):
        return  tf.keras.models.load_model(model_path)

    @staticmethod
    def get_inference(img, level, patch_size, confidence_score=0.75, tissue_threshold=0.5,
                      segmented_results=True):
        prog = 0
        pbar = st.sidebar.empty()
        info = st.sidebar.empty()
        info.write('Testing Constraints')
        p_bar = pbar.progress(prog)
        if 0 > level > 2:
            raise ValueError(f"index_level should be in the range [0, 2].")
        p_bar.progress(prog + 1)
        prog += 1
        if 0 > tissue_threshold > 1:
            raise ValueError(f"tissue_threshold should be in the range [0, 1].")
        p_bar.progress(prog + 1)
        prog += 1
        if 0.5 > confidence_score > 1:
            raise ValueError(f"confidence_score should be in the range [0.5, 1].")
        p_bar.progress(prog + 1)
        prog += 1
        assert len(patch_size) == 2, "patch_size should be a tupple with length = 2."
        p_bar.progress(prog + 1)
        prog += 1

        model = PANDASegmentation.__get_model("./app/models/PANDA_pseudo_segmentation None.h5")

        info.empty()
        info.write('Accessing the image to read')
        img_seg = img.read_region((0, 0), img.level_count - 1, img.level_dimensions[-1])
        img_seg = img_seg.convert('RGB')
        img_seg = np.asarray(img_seg, np.float32) / 255.0

        down_sampling_scale = {level: int(scaling_factor) for level, scaling_factor in enumerate(img.level_downsamples)}

        width, height = img.level_dimensions[level][0] // patch_size[0], img.level_dimensions[level][1] // patch_size[1]

        scale_down_factor = down_sampling_scale[img.level_count - 1 - level]
        scale_up_factor = down_sampling_scale[level]

        tissue_regions = []
        tissue_regions_ind = []


        p_bar.progress(prog + 16)
        prog += 16

        info.empty()
        info.write('Extracting the patches')

        for yidx in range(height):
            for xidx in range(width):
                img_data = img.read_region(
                    (xidx * patch_size[0] * scale_up_factor, yidx * patch_size[1] * scale_up_factor),
                    level, patch_size)
                img_data = img_data.convert('RGB')
                img_data = np.asarray(img_data, np.float32) / 255.0

                unique = np.unique(img_data, return_counts=True)

                if unique[0][0] < 0.93 and np.sum(np.unique(img_data[img_data > 0.93], return_counts=True)[1]) / (
                        patch_size[0] * patch_size[1] * 3) < (1 - tissue_threshold):
                    tissue_regions.append(img_data)
                    tissue_regions_ind.append((yidx, xidx))

        img.close()
        tissue_regions = tf.stack(tissue_regions)

        p_bar.progress(prog + 40)
        prog += 40

        info.empty()
        info.write('Getting Inference')

        preds = model.predict(tissue_regions)
        confidence_matrix = preds.copy()
        confidence_matrix[confidence_matrix >= confidence_score] = 1
        confidence_matrix[confidence_matrix < confidence_score] = 0

        p_bar.progress(prog + 20)
        prog += 20

        info.empty()
        info.write('Generating Segmented map internally')

        preds = np.argmax(confidence_matrix, axis=1)
        img_seg = np.squeeze(img_seg)
        copy_img_seg = img_seg.copy()
        diagnosis = 'Benign'

        if np.sum(preds > 0):
            diagnosis = 'Malignant'

        count = 0
        if segmented_results:
            for ind, tissue_ind in enumerate(tissue_regions_ind):
                if preds[ind] == 1:
                    count += 1
                    roi = img_seg[tissue_ind[0] * patch_size[1] // scale_down_factor: (tissue_ind[0] + 1) * patch_size[
                        1] // scale_down_factor,
                          tissue_ind[1] * patch_size[0] // scale_down_factor: (tissue_ind[1] + 1) * patch_size[
                              0] // scale_down_factor, :]
                    roi[roi[:, :, 0] < 0.93] = (1, 0, 0)

        with st.expander("Results", True):

            col11, col21, col31 = st.columns(3)
            col11.metric("Diagnosis", diagnosis)
            col21.metric("Patches Observed", tissue_regions.shape[0], f'{tissue_threshold * 100}%')
            col31.metric("Potential Malignant Patches", count, f"{count / tissue_regions.shape[0] * 100}%")


            imgs = st.columns(2)

            imgs[0].header('Original Slide Image')
            imgs[0].image(copy_img_seg)

            imgs[1].header('Segmented Results by the model')
            imgs[1].image(img_seg)



        p_bar.progress(prog + 20)
        prog += 20

        info.empty()
        pbar.empty()
        info.success("Done!")
