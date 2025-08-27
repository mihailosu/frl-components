import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

def validate_autoencoder(model, X, y):

    model_out = model(X)

    if isinstance(model_out, tuple):
        out_y, _ = model_out
    else:
        out_y = model_out    

    # Compute the element-wise difference
    diff = np.subtract(out_y, X)

    # Sum absolute differences per-row (per input instance)
    diff = np.sum(np.abs(diff), axis=1)

    # Normalize differences to the range [0, 1]
    min_max_diff = np.max(diff) - np.min(diff)

    diff_norm = (diff - np.min(diff)) / min_max_diff

    # Calculate AUC ROC score
    auc_score = roc_auc_score(y, diff_norm)

    validation_ret = iterate_thresholds(diff_norm, y)

    validation_ret["roc_auc"] = auc_score
    
    return validation_ret


def iterate_thresholds(model_out, y):

    precision, recall, thresholds = precision_recall_curve(y, model_out)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return {
        "precision": precision[best_idx],
        "recall": recall[best_idx],
        "f1": best_f1,
        "threshold": best_threshold
    }


def persist_validation_results(results, experiment_name):
    import csv, os, datetime

    root_path = os.getcwd()

    if experiment_name is not None:
        final_path = os.path.join(root_path, "out", experiment_name)
    else:
        final_path = os.path.join(root_path, "out", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(final_path):
            os.makedirs(final_path)     

    path = os.path.join(final_path, f"validation.csv")

    with open(path, 'w') as f:
         
        writer = csv.writer(f)

        writer.writerow(results[0].keys())

        for res in results:
            writer.writerow(res.values())