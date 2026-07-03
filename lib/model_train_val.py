import torch
import numpy as np
import pickle as pkl
from config.params import *
from lib.stats import *
import warnings

# Model training for AU intensity estimation
def train_model_AU_intensity_estimation(model, train_dataset, train_loader, criterion, optimizer, reg_weights_table, class_weights_table, loss_weights, device, write_results=False, results_path=None):
    model.train()

    all_paths = []
    
    running_regression_loss_mse = 0.0
    running_regression_loss_cos = 0.0
    running_classification_loss = 0.0
    all_pred_labels_regression = []
    all_pred_labels_classification = []
    all_true_labels = []
    
    for paths, images, labels in train_loader:
        all_paths.extend(paths)
        
        this_batch_size = labels.shape[0]
        n_AUs = labels.shape[1]
        
        inputs = images.to(device)
        labels = labels.to(device)
        labels_classification = (labels.unsqueeze(-1) >= torch.arange(1, MAX_INTENSITY + 1).to(device)).float()

        # Organize the weights for each intensity of each AU
        i_indices, j_indices = torch.meshgrid(torch.arange(labels.shape[0], device=device), torch.arange(labels.shape[1], device=device))
        reg_weights = reg_weights_table[j_indices, labels.long()]

        i_indices, j_indices, k_indices = torch.meshgrid(
            torch.arange(labels_classification.shape[0], device=device), 
            torch.arange(labels_classification.shape[1], device=device), 
            torch.arange(labels_classification.shape[2], device=device)
        )
        class_weights = class_weights_table[j_indices, k_indices, labels_classification.long()]

        optimizer.zero_grad()

        outputs = model(inputs)
        
        # Compute the loss
        regression_loss_mse, regression_loss_cos, classification_loss = criterion(outputs, labels, reg_weights, class_weights)
        total_loss = loss_weights[0] * regression_loss_mse + loss_weights[1] * regression_loss_cos + loss_weights[2] * classification_loss
        total_loss.backward()
        optimizer.step()

        running_regression_loss_mse += (regression_loss_mse.item() * this_batch_size)
        running_regression_loss_cos += (regression_loss_cos.item() * this_batch_size)
        running_classification_loss += (classification_loss.item() * this_batch_size)

        outputs_regression = outputs[:, :n_AUs]
        pred_labels_regression = outputs_regression.data.cpu().numpy()
        outputs_classification = outputs[:,n_AUs:]
        pred_labels_classification = sigmoid(outputs_classification.data.cpu().numpy().reshape([this_batch_size, n_AUs, MAX_INTENSITY]))
        pred_labels_classification = pred_labels_classification.sum(axis=-1)
        true_labels = labels.data.cpu().numpy()
        
        all_pred_labels_regression.append(pred_labels_regression)
        all_pred_labels_classification.append(pred_labels_classification)
        all_true_labels.append(true_labels)

    mse_loss = running_regression_loss_mse / len(train_dataset)
    cos_loss = running_regression_loss_cos / len(train_dataset)
    class_loss = running_classification_loss / len(train_dataset)

    pred_labels_regression = np.concatenate(all_pred_labels_regression)
    pred_labels_classification = np.concatenate(all_pred_labels_classification)
    true_labels = np.concatenate(all_true_labels)
    
    iccs_regression = compute_icc_3_1(pred_labels_regression, true_labels)
    iccs_classification = compute_icc_3_1(pred_labels_classification, true_labels)
    
    # Mask out the invalid instances
    mask = true_labels >= 0 
    masked_pred_labels_regression = np.where(mask, pred_labels_regression, np.nan)
    masked_pred_labels_classification = np.where(mask, pred_labels_classification, np.nan)
    masked_true_labels = np.where(mask, true_labels, np.nan)

    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        
        mses_regression = np.nanmean((masked_pred_labels_regression - masked_true_labels) ** 2, axis=0)
        maes_regression = np.nanmean(np.abs(masked_pred_labels_regression - masked_true_labels), axis=0)

        mses_classification = np.nanmean((masked_pred_labels_classification - masked_true_labels) ** 2, axis=0)
        maes_classification = np.nanmean(np.abs(masked_pred_labels_classification - masked_true_labels), axis=0)


    print('Training Results:')
    print("Regression Loss (MSE): {:.3f}".format(mse_loss))
    print("Regression Loss (Cosine): {:.3f}".format(cos_loss))
    print("Classification Loss: {:.3f}".format(class_loss))
    print("ICCs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_regression)))
    print("MAEs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in maes_regression)))
    print("ICCs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_classification)))
    print("MAEs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in maes_classification)))
    print('\n')

    # Save the results
    if write_results == True:
        results = {'image_paths': all_paths,
                   'pred_labels_regression': pred_labels_regression,
                   'pred_labels_classification': pred_labels_classification,
                   'true_labels': true_labels}
        with open(results_path, 'wb') as f:
            pkl.dump(results, f)

    return mse_loss, cos_loss, class_loss, iccs_regression, maes_regression, iccs_classification, maes_classification

# Model validation for AU intensity estimation
def val_model_AU_intensity_estimation(model, val_dataset, val_loader, criterion, reg_weights_table, class_weights_table, device, write_results=False, results_path=None):
    model.eval()

    all_paths = []
    
    running_regression_loss_mse = 0.0
    running_regression_loss_cos = 0.0
    running_classification_loss = 0.0
    all_pred_labels_regression = []
    all_pred_labels_classification = []
    all_true_labels = []

    with torch.no_grad():
        for paths, images, labels in val_loader:
            all_paths.extend(paths)
            
            this_batch_size = labels.shape[0]
            n_AUs = labels.shape[1]
            
            inputs = images.to(device)
            labels = labels.to(device)
            labels_classification = (labels.unsqueeze(-1) >= torch.arange(1, MAX_INTENSITY + 1).to(device)).float()

            # Organize the weights for each intensity of each AU
            i_indices, j_indices = torch.meshgrid(torch.arange(labels.shape[0], device=device), torch.arange(labels.shape[1], device=device))
            reg_weights = reg_weights_table[j_indices, labels.long()]

            i_indices, j_indices, k_indices = torch.meshgrid(
                torch.arange(labels_classification.shape[0], device=device), 
                torch.arange(labels_classification.shape[1], device=device), 
                torch.arange(labels_classification.shape[2], device=device)
            )
            class_weights = class_weights_table[j_indices, k_indices, labels_classification.long()]

            outputs = model(inputs)

            # Compute the loss
            regression_loss_mse, regression_loss_cos, classification_loss = criterion(outputs, labels, reg_weights, class_weights)
            running_regression_loss_mse += regression_loss_mse.item() * this_batch_size
            running_regression_loss_cos += regression_loss_cos.item() * this_batch_size
            running_classification_loss += classification_loss.item() * this_batch_size
            
            outputs_regression = outputs[:, :n_AUs]
            pred_labels_regression = outputs_regression.data.cpu().numpy()
            outputs_classification = outputs[:,n_AUs:]
            pred_labels_classification = sigmoid(outputs_classification.data.cpu().numpy().reshape([this_batch_size, n_AUs, MAX_INTENSITY]))
            pred_labels_classification = pred_labels_classification.sum(axis=-1)
            true_labels = labels.data.cpu().numpy()

            all_pred_labels_regression.append(pred_labels_regression)
            all_pred_labels_classification.append(pred_labels_classification)
            all_true_labels.append(true_labels)

    mse_loss = running_regression_loss_mse / len(val_dataset)
    cos_loss = running_regression_loss_cos / len(val_dataset)
    class_loss = running_classification_loss / len(val_dataset)

    pred_labels_regression = np.concatenate(all_pred_labels_regression)
    pred_labels_classification = np.concatenate(all_pred_labels_classification)
    true_labels = np.concatenate(all_true_labels)
    
    iccs_regression = compute_icc_3_1(pred_labels_regression, true_labels)
    iccs_classification = compute_icc_3_1(pred_labels_classification, true_labels)

    # Mask out the invalid instances
    mask = true_labels >= 0 
    masked_pred_labels_regression = np.where(mask, pred_labels_regression, np.nan)
    masked_pred_labels_classification = np.where(mask, pred_labels_classification, np.nan)
    masked_true_labels = np.where(mask, true_labels, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        
        mses_regression = np.nanmean((masked_pred_labels_regression - masked_true_labels) ** 2, axis=0)
        maes_regression = np.nanmean(np.abs(masked_pred_labels_regression - masked_true_labels), axis=0)

        mses_classification = np.nanmean((masked_pred_labels_classification - masked_true_labels) ** 2, axis=0)
        maes_classification = np.nanmean(np.abs(masked_pred_labels_classification - masked_true_labels), axis=0)

    print("Validation Results:")
    print("Regression Loss (MSE): {:.3f}".format(mse_loss))
    print("Regression Loss (Cosine): {:.3f}".format(cos_loss))
    print("Classification Loss: {:.3f}".format(class_loss))
    print("ICCs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_regression)))
    print("MAEs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in maes_regression)))
    print("ICCs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_classification)))
    print("MAEs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in maes_classification)))
    print('\n')

    # Save the results
    if write_results == True:
        results = {'image_paths': all_paths,
                   'pred_labels_regression': pred_labels_regression,
                   'pred_labels_classification': pred_labels_classification,
                   'true_labels': true_labels}
        with open(results_path, 'wb') as f:
            pkl.dump(results, f)

    return mse_loss, cos_loss, class_loss, iccs_regression, maes_regression, iccs_classification, maes_classification


# Model training for AU detection
def train_model_AU_detection(model, train_dataset, train_loader, criterion, optimizer, weights_table, device, write_results=False, results_path=None):
    model.train()

    all_paths = []
    
    running_loss = 0.0
    all_pred_labels = []
    all_true_labels = []
    
    for paths, images, labels in train_loader:
        all_paths.extend(paths)
        
        this_batch_size = labels.shape[0]
        
        inputs = images.to(device)
        labels = labels.to(device)

        # Organize the weights for each AU
        i_indices, j_indices = torch.meshgrid(torch.arange(labels.shape[0], device=device), torch.arange(labels.shape[1], device=device))
        weights = weights_table[j_indices, labels.long()]

        optimizer.zero_grad()

        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        mask = labels >= 0
        loss_masked = mask * loss
        weighted_loss = weights * loss_masked
        mean_loss = weighted_loss.mean()
        
        mean_loss.backward()
        optimizer.step()

        running_loss += (mean_loss.item() * this_batch_size)

        pred_labels = outputs.data.cpu().numpy()
        true_labels = labels.data.cpu().numpy()
        
        all_pred_labels.append(pred_labels)
        all_true_labels.append(true_labels)

    loss = running_loss / len(train_dataset)

    pred_labels = np.concatenate(all_pred_labels)
    true_labels = np.concatenate(all_true_labels)

    # Mask out the invalid instances
    mask = true_labels >= 0
    masked_pred_labels = np.where(mask, pred_labels, np.nan)
    masked_true_labels = np.where(mask, true_labels, np.nan)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        accs = np.nanmean((masked_pred_labels >= 0.0) == masked_true_labels, axis=0)

    print('Training Results:')
    print("Loss: {:.3f}".format(loss))
    print("Accs: [{}]".format(', '.join(f'{i:.3f}' for i in accs)))
    print('\n')

    # Save the results
    if write_results == True:
        results = {'image_paths': all_paths,
                   'pred_labels': pred_labels,
                   'true_labels': true_labels}
        with open(results_path, 'wb') as f:
            pkl.dump(results, f)

    return loss, accs

# Model validation for AU detection
def val_model_AU_detection(model, val_dataset, val_loader, criterion, weights_table, device, write_results=False, results_path=None):
    model.eval()

    all_paths = []
    
    running_loss = 0.0
    all_pred_labels = []
    all_true_labels = []
    
    with torch.no_grad():
        for paths, images, labels in val_loader:
            all_paths.extend(paths)
            
            this_batch_size = labels.shape[0]
            
            inputs = images.to(device)
            labels = labels.to(device)
    
            # Organize the weights for each AU
            i_indices, j_indices = torch.meshgrid(torch.arange(labels.shape[0], device=device), torch.arange(labels.shape[1], device=device))
            weights = weights_table[j_indices, labels.long()]

            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            mask = labels >= 0
            loss_masked = mask * loss
            weighted_loss = weights * loss
            mean_loss = weighted_loss.mean()
    
            running_loss += (mean_loss.item() * this_batch_size)
    
            pred_labels = outputs.data.cpu().numpy()
            true_labels = labels.data.cpu().numpy()
            
            all_pred_labels.append(pred_labels)
            all_true_labels.append(true_labels)
    
    loss = running_loss / len(val_dataset)

    pred_labels = np.concatenate(all_pred_labels)
    true_labels = np.concatenate(all_true_labels)
    
    # Mask out the invalid instances
    mask = true_labels >= 0
    masked_pred_labels = np.where(mask, pred_labels, np.nan)
    masked_true_labels = np.where(mask, true_labels, np.nan)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        accs = np.nanmean((masked_pred_labels >= 0.0) == masked_true_labels, axis=0)

    print("Validation Results:")
    print("Loss: {:.3f}".format(loss))
    print("Accs: [{}]".format(', '.join(f'{i:.3f}' for i in accs)))
    print('\n')

    # Save the results
    if write_results == True:
        results = {'image_paths': all_paths,
                   'pred_labels': pred_labels,
                   'true_labels': true_labels}
        with open(results_path, 'wb') as f:
            pkl.dump(results, f)

    return loss, accs