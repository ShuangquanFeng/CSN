import torch
from config.params import *
def compute_frequency_intensity(all_labels, device): # Compute the frequency of each intensity of each AU in the dataset
    n_AUs = len(all_labels[0])
    intensity_counts = torch.zeros((n_AUs, MAX_INTENSITY + 1), dtype=torch.int, device=device)

    for labels in all_labels:
        for i in range(n_AUs):
            intensity = labels[i]
            if intensity >= 0:
                intensity_counts[i, int(intensity)] += 1
    
    threshold_counts = torch.zeros((n_AUs, MAX_INTENSITY, 2), dtype=torch.int, device=device)

    for i in range(n_AUs):
        for j in range(1, MAX_INTENSITY + 1):
            threshold_counts[i, j-1, 0] = torch.sum(intensity_counts[i, :j])
            threshold_counts[i, j-1, 1] = torch.sum(intensity_counts[i, j:])

    return intensity_counts, threshold_counts

def compute_weights_intensity(counts, method, device, groups = None): # Compute the weights of each intensity of each AU for the dataset
    counts_grouped = counts.clone()
    if groups != None:
        for group in groups:
            counts_grouped[:,group] = torch.sum(counts[:,group], axis=1, keepdim=True, dtype=counts.dtype)
    weights = torch.zeros_like(counts_grouped, dtype=torch.float, device=device)
    if method == 'uniform':
        weights[counts_grouped > 0] = 1
    elif method == 'inverse':
        weights[counts_grouped > 0] = 1 / counts_grouped[counts_grouped > 0]
    elif method == 'inverse_sqrt':
        weights[counts_grouped > 0] = 1 / np.sqrt(counts_grouped[counts_grouped > 0])
    elif method == 'inverse_log':
        counts[counts_grouped > 0] += 1
        weights[counts_grouped > 0] = 1 / np.log(counts_grouped[counts_grouped > 0])
    else:
        raise ValueError("Unknown weight option. It should be 'uniform', 'inverse', 'inverse_sqrt', or 'inverse_log'.")
    final_weights = torch.zeros_like(weights, dtype=torch.float, device=device)
    final_weights = weights / weights.sum(dim=tuple(range(1, final_weights.dim())), keepdim=True)
    return final_weights

def compute_frequency_occurrence(all_labels, device): # Compute the frequency of the occurrences of each AU in the dataset
    n_AUs = len(all_labels[0])
    label_counts = torch.zeros((n_AUs, 2), dtype=torch.int, device=device)
    for labels in all_labels:
        for i in range(n_AUs):
            label = labels[i]
            if label >= 0:
                label_counts[i, int(label)] += 1
    return label_counts

def compute_weights_occurrence(counts, method, device): # Compute the weights of the occurrences of each AU for the dataset
    counts_grouped = counts.clone()
    weights = torch.zeros_like(counts, dtype=torch.float, device=device)
    if method == 'uniform':
        weights[counts_grouped > 0] = 1
    elif method == 'inverse':
        weights[counts_grouped > 0] = 1 / counts_grouped[counts_grouped > 0]
    elif method == 'inverse_sqrt':
        weights[counts_grouped > 0] = 1 / np.sqrt(counts_grouped[counts_grouped > 0])
    elif method == 'inverse_log':
        counts[counts_grouped > 0] += 1
        weights[counts_grouped > 0] = 1 / np.log(counts_grouped[counts_grouped > 0])
    else:
        raise ValueError("Unknown weight option. It should be 'uniform', 'inverse', 'inverse_sqrt', or 'inverse_log'.")
    final_weights = torch.zeros_like(weights, dtype=torch.float, device=device)
    final_weights = weights / weights.sum(dim=tuple(range(1, final_weights.dim())), keepdim=True)
    return final_weights