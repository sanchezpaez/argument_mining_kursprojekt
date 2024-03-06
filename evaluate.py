import numpy as np
from sklearn.metrics import classification_report


def generate_classification_report(actual_labels_numeric, predicted_labels_numeric, all_labels):
    """Generate and print classification report"""
    class_names = all_labels
    print("Classification Report:")
    report = classification_report(actual_labels_numeric, predicted_labels_numeric,
                                   labels=np.arange(0, len(all_labels)), target_names=class_names, digits=4,
                                   zero_division=0)
    print(report)
    return report
