import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from settings import FEATURE_CSV_FILE, LABEL_CSV_FILE, CLASSIFICATION_MODEL


class ClassifierTrainer:
    def __init__(self):
        self.x_data = pd.read_csv(FEATURE_CSV_FILE, header=None)
        self.y_data = pd.read_csv(LABEL_CSV_FILE)

        self.model_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                            "Naive Bayes", "QDA"]
        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=2, degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                decision_function_shape='ovr', random_state=None),
            SVC(gamma=2, C=1, probability=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    def train_best_model(self, model_name):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.x_data, self.y_data, test_size=.3, random_state=42)
        best_clf = self.classifiers[self.model_names.index(model_name)]
        best_clf.fit(self.x_data, self.y_data)
        import numpy as np
        pred_res = best_clf.predict_proba(np.array(self.x_data[:, 700]).reshape(1, -1))
        matched_index = np.argmax(pred_res)
        label = best_clf.classes_[matched_index]
        print(label)
        score = best_clf.score(x_test, y_test)
        print(score)
        joblib.dump(best_clf, CLASSIFICATION_MODEL)
        print(f"Successfully saved in {CLASSIFICATION_MODEL}")

        return

    def test_several_models(self):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.x_data, self.y_data, test_size=.3, random_state=42)

        for name, clf in zip(self.model_names, self.classifiers):
            clf.fit(self.x_data, self.y_data)
            score = clf.score(x_test, y_test)
            print(f"model:{name}, score:{score}")

        return


if __name__ == '__main__':
    ClassifierTrainer().train_best_model(model_name="RBF SVM")
