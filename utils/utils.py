import torch


def calculate_class_weights_from_dataframe(df, target_column, label_maps):
    targets = df[target_column].values
    labels = list(label_maps)
    class_counts = torch.bincount(torch.tensor([labels.index(target) for target in targets]))
    total_samples = class_counts.sum().float()
    class_weights = total_samples / (class_counts.float() + 1e-6)
    class_weights /= class_weights.sum()
    return class_weights
