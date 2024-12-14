# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
import os 
import argparse
import yaml
from termcolor import colored
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold
from models import DepDet, DepressionDetector, DepressionDet
from datasets import get_dvlog_dataloader
from utils.viz import plot_tsne, improve_tsne_visualization  
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold

def collate_fn(batch):
    # Assuming x is the data and y is the label
    data, labels = zip(*batch)
    # Example: pad data to max length in the batch
    max_length = max(d.shape[0] for d in data)
    padded_data = [np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant') for d in data]
    return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def LOG_INFO(msg, mcolor='blue'):
    '''Print log messages with color.'''
    print(colored("#LOG :", 'green') + colored(msg, mcolor)) # type: ignore

# Setting seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2024)

CONFIG_PATH = "./config.yaml"

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Train and test a model on the DVLOG dataset.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument("-m", "--model", type=str, choices=["DepressionDetector", "DepDet"])
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-sch", "--lr_scheduler", type=str, choices=["cos", "None",])
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of folds for cross-validation")
    parser.set_defaults(**config)
    args = parser.parse_args()
    return args

def extract_features(net, loader, device):
    """Extract features from the intermediate layer of the model."""
    net.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        with tqdm(loader, desc="Extracting features", leave=False, unit="batch") as pbar:
            for x, y in pbar:
                x = x.to(device)
                z, _, _ = net.feature_extractor(x)  # Extract features
                all_features.append(z.cpu().numpy())
                all_labels.append(y.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels

###----------------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    LOG_INFO(args)

    # Prepare data for k-fold cross-validation
    train_loader = get_dvlog_dataloader(args.data_dir, "train", args.batch_size, args.train_gender)
    val_loader = get_dvlog_dataloader(args.data_dir, "valid", args.batch_size, args.test_gender)
    test_loader = get_dvlog_dataloader(args.data_dir, "test", args.batch_size, args.test_gender)

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(combined_dataset))

    # Variables to hold features and labels for all folds
    features_dict = {}  # Dictionary to store features for each sample ID
    labels_dict = {}  # Dictionary to store labels for each sample ID

    for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        LOG_INFO(f"Fold {fold+1}/{args.num_folds}")

        train_subset = Subset(combined_dataset, train_indices.tolist())
        val_subset = Subset(combined_dataset, val_indices.tolist())

        train_loader_fold = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader_fold = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        # Construct the model
        if args.model == "DepDet":
            net = DepDet(d=256, l=6) 
        elif args.model == "DepressionDetector":
            net = DepressionDetector(d=256, l=6, t_downsample=4)
        elif args.model == "DepressionDet":
            net = DepressionDet(d=256, l=6, t_downsample=4)

        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        LOG_INFO(f"[{args.model}] Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}", "cyan")
        
        # Load the best saved model for the current fold
        # model_path = os.path.join("./weights", f"best_model_wo_lrs_{fold}.pt")
        model_path = os.path.join("./weights", f"best_model_wo_lrs_3.pt") 
        # model_path = os.path.join("./weights", f"best_model_wo_lrs_5.pt") 
        # model_path = os.path.join("./weights", f"averaged_model_weights.pt")

        net.load_state_dict(torch.load(model_path, map_location=args.device[0]))

        # Extract features from the validation dataset of the current fold
        LOG_INFO(f"Extracting features for Fold {fold+1}")
        fold_features, fold_labels = extract_features(net, val_loader_fold, args.device[0])

        # Accumulate features and labels in a dictionary (by sample index)
        for i, feature in enumerate(fold_features):
            sample_id = val_indices[i]  # Use the validation index as a unique identifier
            if sample_id not in features_dict:
                features_dict[sample_id] = []
                labels_dict[sample_id] = fold_labels[i]  # Assuming labels are consistent across folds

            # Append the current fold's features for this sample ID
            features_dict[sample_id].append(feature)

    # Calculate the average features for each sample across all folds
    average_features = []
    average_labels = []

    for sample_id, feature_list in features_dict.items():
        # Calculate the average of features across all folds
        avg_feature = np.mean(feature_list, axis=0)
        average_features.append(avg_feature)

        # Labels are consistent across folds, so we just take the label from labels_dict
        average_labels.append(labels_dict[sample_id])

    # Convert to numpy arrays for further processing
    average_features = np.array(average_features)
    average_labels = np.array(average_labels)

    # Plot t-SNE using averaged features and labels
    LOG_INFO("Plotting t-SNE with averaged fold features...")
    plot_tsne(average_features, average_labels, title="t-SNE Plot of Averaged Features across All Folds", filename="t_sne_avg.png")

    # Call the function to improve t-SNE visualization
    improve_tsne_visualization(average_features, average_labels, use_umap=False, tsne_perplexity=40, tsne_learning_rate=200, filename="_improve_t_sne_avg.png")

if __name__ == "__main__":
    main()
