import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics
path_to_output = "Output/2class"

def get_all_results():
    # sub_list = [2, 6, 7, 8, 11, 12, 18, 20, 21, 25]
    for subidx in range(1,26):
    # for subidx in sub_list:
        print(f"Subject {subidx}")
        result_fname = f"test_base_{subidx}.pickle"
        with (open(os.path.join(path_to_output, result_fname), "rb")) as file:
            test_result = pickle.load(file)
            print(test_result)

def get_confusion_matrix(sub_idx, n_classes):
    plt.tight_layout()
    path_to_output = os.path.join("Output", f"{n_classes}class")
    result_fname = f"test_base_{sub_idx}.pickle"
    UNIQUE_LABELS = ["all", "Backward", "Cylindrical", "Down", "Forward", "Left", "Lumbrical", "Right", "Spherical",
                     "Up", "twist_Left", "twist_Right"]
    labels_idx_for_classif = [2, 4, 10, 11] if n_classes == 4 else list(range(1,12))
    labels_for_classif = np.array([UNIQUE_LABELS[idx] for idx in labels_idx_for_classif])
    with (open(os.path.join(path_to_output, result_fname), "rb")) as file:
        test_result = pickle.load(file)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=test_result['conf_matr'],
                                                      display_labels=labels_for_classif)
        disp.plot()
        disp.figure_.autofmt_xdate(rotation=90)
        disp.figure_.show()



if __name__ == "__main__":
    get_all_results()
    # get_confusion_matrix(sub_idx=20, n_classes=2)