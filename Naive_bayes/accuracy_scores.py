def accuracy_score(y_true, y_pred):
    """Compute simple accuracy (fraction correct). Works with lists or arrays."""
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def classification_report_simple(y_true, y_pred):
    """Return per-class precision/recall/support in a dict."""
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    labels = set(y_true) | set(y_pred)
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    report = {}
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        rec = tp[l] / support[l] if support[l] > 0 else 0.0
        report[l] = {"precision": prec, "recall": rec, "support": support[l]}
    return report
