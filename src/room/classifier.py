import joblib
# import time
import numpy as np

from src.feature.extractor import ImageFeature
from settings import CLASSIFICATION_MODEL


class ClassExtractor:
    def __init__(self):
        self.feature_extractor = ImageFeature()
        self.model = joblib.load(CLASSIFICATION_MODEL)
        self.image_dir = '/media/image-share'
        print(self.model.classes_)

    def extract_class(self, origin_frame_path):
        # st_time = time.time()
        frame_feature = self.feature_extractor.get_feature_from_image(img_path=origin_frame_path)
        input_feature = np.array(frame_feature).reshape(1, -1)
        predict_result = self.model.predict_proba(input_feature)
        prob = np.max(predict_result)
        matched_index = np.argmax(predict_result)
        label = self.model.classes_[matched_index]

        # print(f"Class: {label} with Probability: {prob}")
        # print(f"Process Time: {time.time() - st_time}s")

        return label, prob


if __name__ == '__main__':
    classifier = ClassExtractor()
    import glob
    import os
    old_bad_images = glob.glob(os.path.join("", "*.*"))
    for image_path in old_bad_images:
        lbl, _ = classifier.extract_class(origin_frame_path=image_path)
        if lbl == "OldGood":
            print(image_path)
