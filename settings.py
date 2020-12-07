import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'utils', 'model')
DATA_DIR = os.path.join(CUR_DIR, 'data')
TRAIN_DATA_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'train_data'))

FEATURE_CSV_FILE = os.path.join(TRAIN_DATA_DIR, 'feature.csv')
LABEL_CSV_FILE = os.path.join(TRAIN_DATA_DIR, 'label.csv')
CLASSIFICATION_MODEL = os.path.join(MODEL_DIR, 'classifier.pkl')

INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
