import argparse
import torch
import torch.optim as optim
import numpy as np
import pickle as pkl
import os
import random
import time
from config.params import *
from config.datasets import *
from config.models import *
from lib.compute_weights import *
from lib.model_train_val import *
from lib.transforms_compose import *
from data.AU_dataset import *
from loss.combination_loss import *
from models.iresnet_Siamese import *

# sample bash command: 
# python DISFA_AU_detection_threefold_CSN-IR50.py --merge_locs stage4 --epochs 3 --batch_size 64 --gpu 0 --save_interval 1

parser = argparse.ArgumentParser(description='Evaluate the CSN-IR50 model on the DISFA dataset for AU detection using threefold cross-validation.')
parser.add_argument('--merge_locs', type=str, nargs='+', help="the location(s) where the difference(s) of the two networks are computed for feature extraction", required=True)
parser.add_argument('--epochs', default=N_EPOCHS, type=int, help="number of epochs for training")
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--gpu', default=0, type=int, help="the GPU to use")
parser.add_argument('--save_interval', default=1, type=int, help='define the interval of epochs to save model state')

def main():
    sel_AUs = DISFA_AUS_D.copy()
    n_AUs = len(sel_AUs)
    
    # Make the model training deterministic
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parser.parse_args()

    merge_locs = args.merge_locs
    merg_locs_str = ','.join(merge_locs)

    num_epochs = args.epochs
    batch_size = args.batch_size
    interval = args.save_interval

    task = "DISFA_AU_detection_threefold"
    
    data_transforms = {}
    data_transforms['train'] = transforms_compose(TRANSFORM_STEPS, include_random=True)
    data_transforms['val'] = transforms_compose(TRANSFORM_STEPS, include_random=False)

    device = torch.device('cuda:' + str(args.gpu))
    
    for fold in range(1, 4):
        print('Fold ' + str(fold))
        
        # Create the folders for saving results
        models_folder = os.path.join('results', task, f"CSN-{MODEL_BACKBONE}-{merg_locs_str}", 'trained_models', 'fold-' + str(fold))
        metrics_folder = os.path.join('results', task, f"CSN-{MODEL_BACKBONE}-{merg_locs_str}", 'metrics', 'fold-' + str(fold))
        results_folder = os.path.join('results', task, f"CSN-{MODEL_BACKBONE}-{merg_locs_str}", 'results', 'fold-' + str(fold))
        os.makedirs(models_folder, exist_ok=True)
        os.makedirs(metrics_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        metrics_path = os.path.join(metrics_folder, 'metrics.pkl')
        
        val_subs = DISFA_3FOLDS[fold-1]
        train_subs = [sub for sub in DISFA_SUBJECTS if sub not in val_subs]
    
        # Initialize the model
        print("Initializing the model...")
        model = iresnet_Siamese(n_layers=N_LAYERS, n_features = n_AUs, merge_locs = merge_locs, pretrained = True, weights_path = FACE_RECOGNITION_MODEL_PATH)
        model = model.to(device)
        torch.cuda.device(args.gpu)
        
        # Initialize the dataset and dataloader
        print("Initializing Datasets and Dataloaders...")
        train_dataset = AU_dataset_frame_pairs(subject_dict = {'DISFAleft': train_subs, 'DISFAright': train_subs}, transform = data_transforms['train'], label_type = "binary occurrence")
        val_dataset = AU_dataset_frame_pairs(subject_dict = {'DISFAleft': val_subs, 'DISFAright': val_subs}, transform = data_transforms['val'], label_type = "binary occurrence")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Model training settings
        last_layer = list(model.children())[-1]
        ignored_params = list(map(id, last_layer.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.parameters())
        optimizer = optim.Adam([{'params': base_params, 'lr': OPTIMIZER_SETTINGS['base_lr']},
                                {'params': last_layer.parameters(), 'lr': OPTIMIZER_SETTINGS['last_layer_lr']}], weight_decay = OPTIMIZER_SETTINGS['weight_decay'])

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Compute the weights of different labels
        print("Computing weights...")
        label_counts = compute_frequency_occurrence(train_dataset.all_labels, device)  
        weights_table = compute_weights_occurrence(label_counts, CLASS_WEIGHT_METHOD, device)
        
        # Train and validate the model
        metrics = []
        for epoch in range(1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            
            since = time.time()
            
            if epoch % interval == 0:
                write_results = True
                train_results_path = os.path.join(results_folder, 'training_results_epoch' + str(epoch) + '.pkl')
                val_results_path = os.path.join(results_folder, 'val_results_epoch' + str(epoch) + '.pkl')
            else:
                write_results = False
                train_results_path = None
                val_results_path = None                        

            train_loss, train_accs = train_model_AU_detection(model, train_dataset, train_loader, criterion, optimizer, weights_table, device, write_results=write_results, results_path=train_results_path)

            val_loss, val_accs = val_model_AU_detection(model, val_dataset, val_loader, criterion, weights_table, device, write_results=write_results, results_path=val_results_path)

            epoch_metrics = {
                'epoch': epoch,
                'train': {
                    'loss': train_loss,
                    'accs': train_accs
                },
                'val': {
                    'loss': val_loss,
                    'accs': val_accs
                }
            }
            metrics.append(epoch_metrics)
            with open(metrics_path, 'wb') as f:
                pkl.dump(metrics, f)

            if epoch % interval == 0:
                torch.save(model.state_dict(), os.path.join(models_folder, f'checkpoint_epoch{epoch}.pth'))
    
            time_elapsed = time.time() - since
            print('Epoch {} complete in {:.0f}m {:.0f}s\n'.format(epoch, time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    main()