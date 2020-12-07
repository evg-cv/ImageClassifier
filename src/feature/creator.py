import os
import glob
import csv
import pandas as pd

from src.feature.extractor import ImageFeature
from settings import DATA_DIR, FEATURE_CSV_FILE, LABEL_CSV_FILE


class TrainFeatureCreator:
    def __init__(self):
        self.image_feature_extractor = ImageFeature()

    @staticmethod
    def create_csv_file(file_path, content):
        with open(file_path, 'w', newline='') as fp:
            wr = csv.writer(fp, delimiter=',')
            wr.writerows(content)

        print(f"Successfully saved in {file_path}")

        return

    def create_train_feature_data(self):
        subdir = os.listdir(DATA_DIR)
        features = []
        labels = []
        for s_dir in subdir:
            frames_path = glob.glob(os.path.join(DATA_DIR, s_dir, "*.jpg"))
            for f_path in frames_path:
                frame_feature = self.image_feature_extractor.get_feature_from_image(img_path=f_path).tolist()
                features.append(frame_feature)
                labels.append(s_dir)

        print(len(features))

        self.create_csv_file(file_path=FEATURE_CSV_FILE, content=features)
        pd.DataFrame(list(zip(labels)),
                     columns=["Label"]).to_csv(LABEL_CSV_FILE, index=False, header=True, mode="w")

        return


if __name__ == '__main__':
    TrainFeatureCreator().create_train_feature_data()
