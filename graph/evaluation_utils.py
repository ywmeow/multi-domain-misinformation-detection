from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, average_precision_score


def metrics(y, p, probability):
    both = len(set(y)) == 2
    report = classification_report(y,p,labels=[0,1],target_names=['real','fake'],output_dict=True,zero_division=0)
    return {
        'n':len(y),
        'accuracy':float(accuracy_score(y,p)),
        'macro_precision':float(report['macro avg']['precision']),
        'macro_recall':float(report['macro avg']['recall']),
        'macro_f1':float(report['macro avg']['f1-score']),
        'balanced_accuracy':float(balanced_accuracy_score(y,p)) if both else None,
        'fake_f1':float(f1_score(y,p,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,probability)) if both else None,
        'average_precision':float(average_precision_score(y,probability)) if both else None,
        'confusion_matrix_real_fake':confusion_matrix(y,p,labels=[0,1]).tolist(),
        'classification_report':report,
    }
