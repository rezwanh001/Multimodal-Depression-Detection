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
from models import DepDet, DepressionDetector, DepressionDet
from models import FocalLoss, CombinedLoss
from datasets import get_dvlog_dataloader, kfold_get_dvlog_dataloader
from utils import plot_confusion_matrix, plot_confusion_matrix_mean
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold

def collate_fn(batch):
    # Assuming x is the data and y is the label
    data, labels = zip(*batch)
    # Example: pad data to max length in the batch
    max_length = max(d.shape[0] for d in data)
    padded_data = [np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant') for d in data]
    return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG :", 'green') + colored(msg, mcolor)) # type: ignore

# Seed 
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

    parser = argparse.ArgumentParser(
        description="Train and test a model on the DVLOG dataset."
    )
    # arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m", "--model", type=str,
        choices=["DepressionDetector", "DepDet"]
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument(
        "-sch", "--lr_scheduler", type=str,
        choices=["cos", "None",]
    )
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args

def val(
    net, val_loader, loss_fn, device, 
):
    """Test the model on the validation / test set."""
    net.eval()
    sample_count = 0
    running_loss = 0.
    running_gsa_loss = 0.
    running_hatnet_loss = 0.

    # Initialize variables for metrics
    TP, FP, TN, FN = 0, 0, 0, 0
    TP_gsa, FP_gsa, TN_gsa, FN_gsa = 0, 0, 0, 0
    TP_hatnet, FP_hatnet, TN_hatnet, FN_hatnet = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch"
        ) as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device).unsqueeze(1)

                # Pass to Model
                y_pred, gsa_pred, hatnet_pred = net(x)

                ## depdet loss (total loss)
                loss = loss_fn(y_pred, y.to(torch.float32), net)

                ## gsa loss
                loss_gsa = loss_fn(gsa_pred, y.to(torch.float32), net)

                ## hatnet loss  
                loss_hatnet = loss_fn(hatnet_pred, y.to(torch.float32), net) 

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                running_gsa_loss += loss_gsa.item() * x.shape[0]
                running_hatnet_loss += loss_hatnet.item() * x.shape[0]

                # binary classification with only one output neuron
                ## Total Preds
                pred = (y_pred > 0).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                ## GSA Preds
                pred_gsa = (gsa_pred > 0).int()
                TP_gsa += torch.sum((pred_gsa == 1) & (y == 1)).item()
                FP_gsa += torch.sum((pred_gsa == 1) & (y == 0)).item()
                TN_gsa += torch.sum((pred_gsa == 0) & (y == 0)).item()
                FN_gsa += torch.sum((pred_gsa == 0) & (y == 1)).item()

                ## HatNet Preds
                pred_hatnet = (hatnet_pred > 0).int()
                TP_hatnet += torch.sum((pred_hatnet == 1) & (y == 1)).item()
                FP_hatnet += torch.sum((pred_hatnet == 1) & (y == 0)).item()
                TN_hatnet += torch.sum((pred_hatnet == 0) & (y == 0)).item()
                FN_hatnet += torch.sum((pred_hatnet == 0) & (y == 1)).item()

                # Calculate metrics for total loss
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                # Calculate metrics for GSA loss
                gsa_precision = TP_gsa / (TP_gsa + FP_gsa) if (TP_gsa + FP_gsa) > 0 else 0.0
                gsa_recall = TP_gsa / (TP_gsa + FN_gsa) if (TP_gsa + FN_gsa) > 0 else 0.0
                gsa_f1_score = (
                    2 * (gsa_precision * gsa_recall) / (gsa_precision + gsa_recall) 
                    if (gsa_precision + gsa_recall) > 0 else 0.0
                )
                gsa_accuracy = (
                    (TP_gsa + TN_gsa) / sample_count
                    if sample_count > 0 else 0.0
                )

                # Calculate metrics for HatNet loss
                hatnet_precision = TP_hatnet / (TP_hatnet + FP_hatnet) if (TP_hatnet + FP_hatnet) > 0 else 0.0
                hatnet_recall = TP_hatnet / (TP_hatnet + FN_hatnet) if (TP_hatnet + FN_hatnet) > 0 else 0.0
                hatnet_f1_score = (
                    2 * (hatnet_precision * hatnet_recall) / (hatnet_precision + hatnet_recall) 
                    if (hatnet_precision + hatnet_recall) > 0 else 0.0
                )
                hatnet_accuracy = (
                    (TP_hatnet + TN_hatnet) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": running_loss / sample_count, 
                    "loss_gsa": running_gsa_loss / sample_count,
                    "loss_hatnet": running_hatnet_loss / sample_count,
                    "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                    "gsa_acc": gsa_accuracy,
                    "gsa_precision": gsa_precision, "gsa_recall": gsa_recall, "gsa_f1": gsa_f1_score,
                    "hatnet_acc": hatnet_accuracy,
                    "hatnet_precision": hatnet_precision, "hatnet_recall": hatnet_recall, "hatnet_f1": hatnet_f1_score,
                })

    return {
        "loss": running_loss / sample_count, 
        "loss_gsa": running_gsa_loss / sample_count,
        "loss_hatnet": running_hatnet_loss / sample_count,
        "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
        "gsa_acc": gsa_accuracy,
        "gsa_precision": gsa_precision, "gsa_recall": gsa_recall, "gsa_f1": gsa_f1_score,
        "hatnet_acc": hatnet_accuracy,
        "hatnet_precision": hatnet_precision, "hatnet_recall": hatnet_recall, "hatnet_f1": hatnet_f1_score,

        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TP_gsa": TP_gsa, "FP_gsa": FP_gsa, "TN_gsa": TN_gsa, "FN_gsa": FN_gsa,
        "TP_hatnet": TP_hatnet, "FP_hatnet": FP_hatnet, "TN_hatnet": TN_hatnet, "FN_hatnet": FN_hatnet,
    }

def main():
    args = parse_args()
    LOG_INFO(args)

    # Initialize K-Fold cross-validation
    train_loader = get_dvlog_dataloader(args.data_dir, "train", args.batch_size, args.train_gender)
    val_loader = get_dvlog_dataloader(args.data_dir, "valid", args.batch_size, args.test_gender)
    test_loader = get_dvlog_dataloader(args.data_dir, "test", args.batch_size, args.test_gender)

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(combined_dataset))

    ##-----------------------------------------------------------------
    # Initialize accumulators for TP, FP, TN, FN across all 10 folds
    TP_sum, FP_sum, TN_sum, FN_sum = 0, 0, 0, 0
    TP_gsa_sum, FP_gsa_sum, TN_gsa_sum, FN_gsa_sum = 0, 0, 0, 0
    TP_hatnet_sum, FP_hatnet_sum, TN_hatnet_sum, FN_hatnet_sum = 0, 0, 0, 0

    fold_results = []
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
        
        # Set other training components
        loss_fn = CombinedLoss(lambda_reg=1e-5, focal_weight=0.5, l2_weight=0.5)
        
        # Load the best model for testing
        net.load_state_dict(
            torch.load(os.path.join("./weights", f"best_model_wo_lrs_{fold}.pt"), map_location=args.device[0])
            # torch.load(os.path.join("./weights", f"best_model_{fold}.pt"), map_location=args.device[0])
        )
        test_results = val(net, val_loader_fold, loss_fn, args.device[0])

        ## calculate overall (avg) conf & vizualization -----------------------
        # Accumulate confusion matrix values for main model
        TP_sum += test_results["TP"]
        FP_sum += test_results["FP"]
        TN_sum += test_results["TN"]
        FN_sum += test_results["FN"]
        ## plots
        # plot_confusion_matrix(test_results["TP"], test_results["FP"], test_results["TN"], test_results["FN"], 
        #                       title=f"mainkfold_confusion_matrix_{fold}", filename=f"mainkfold_confusion_matrix_{fold}.png")

        # Accumulate confusion matrix values for GSA model
        TP_gsa_sum += test_results["TP_gsa"]
        FP_gsa_sum += test_results["FP_gsa"]
        TN_gsa_sum += test_results["TN_gsa"]
        FN_gsa_sum += test_results["FN_gsa"]
        ## plots
        # plot_confusion_matrix(test_results["TP_gsa"], test_results["FP_gsa"], test_results["TN_gsa"], test_results["FN_gsa"], 
        #                       title=f"mainkfold_gsa_confusion_matrix_{fold}", filename=f"mainkfold_gsa_confusion_matrix_{fold}.png")

        # Accumulate confusion matrix values for HatNet model
        TP_hatnet_sum += test_results["TP_hatnet"]
        FP_hatnet_sum += test_results["FP_hatnet"]
        TN_hatnet_sum += test_results["TN_hatnet"]
        FN_hatnet_sum += test_results["FN_hatnet"]
        ## plots
        # plot_confusion_matrix(test_results["TP_hatnet"], test_results["FP_hatnet"], test_results["TN_hatnet"], test_results["FN_hatnet"], 
        #                       title=f"mainkfold_hatnet_confusion_matrix_{fold}", filename=f"mainkfold_hatnet_confusion_matrix_{fold}.png")
        ##-----------------------------------------------------------------

        fold_results.append(test_results)
        LOG_INFO(f"Fold {fold+1} test results: {fold_results[-1]}", mcolor='yellow')

    ##------------------- avg confusion matrix--------------------------
    # Compute the average confusion matrix for each model
    TP_avg, FP_avg, TN_avg, FN_avg = round(TP_sum / 10), round(FP_sum / 10), round(TN_sum / 10), round(FN_sum / 10)
    TP_gsa_avg, FP_gsa_avg, TN_gsa_avg, FN_gsa_avg = round(TP_gsa_sum / 10), round(FP_gsa_sum / 10), round(TN_gsa_sum / 10), round(FN_gsa_sum / 10)
    TP_hatnet_avg, FP_hatnet_avg, TN_hatnet_avg, FN_hatnet_avg = round(TP_hatnet_sum / 10), round(FP_hatnet_sum / 10), round(TN_hatnet_sum / 10), round(FN_hatnet_sum / 10)

    # Now you can use the previously provided code to plot the averaged confusion matrix
    plot_confusion_matrix(TP_avg, FP_avg, TN_avg, FN_avg, 
                          title="Main Model Averaged Confusion Matrix", filename="mainkfold_main_model_avg_confusion_matrix.png")
    plot_confusion_matrix(TP_gsa_avg, FP_gsa_avg, TN_gsa_avg, FN_gsa_avg, 
                          title="GSA Model Averaged Confusion Matrix", filename="mainkfold_gsa_avg_confusion_matrix.png")
    plot_confusion_matrix(TP_hatnet_avg, FP_hatnet_avg, TN_hatnet_avg, FN_hatnet_avg, 
                          title="HatNet Model Averaged Confusion Matrix", filename="mainkfold_hatnet_avg_confusion_matrix.png")

    ##----------------------------------------

    print("============---------------================")
    LOG_INFO("Overall Folds Results for the test sets")
    LOG_INFO(f"All Results: {fold_results}")

    # Aggregate results across folds
    avg_results = {
        "acc": np.mean([fr["acc"] for fr in fold_results]),
        "precision": np.mean([fr["precision"] for fr in fold_results]),
        "recall": np.mean([fr["recall"] for fr in fold_results]),
        "f1": np.mean([fr["f1"] for fr in fold_results]),

        "loss": np.mean([fr["loss"] for fr in fold_results]),
        "loss_gsa": np.mean([fr.get("loss_gsa", 0) for fr in fold_results]),
        "loss_hatnet": np.mean([fr.get("loss_hatnet", 0) for fr in fold_results]),

        "gsa_acc": np.mean([fr.get("gsa_acc", 0) for fr in fold_results]),
        "gsa_precision": np.mean([fr.get("gsa_precision", 0) for fr in fold_results]),
        "gsa_recall": np.mean([fr.get("gsa_recall", 0) for fr in fold_results]),
        "gsa_f1": np.mean([fr.get("gsa_f1", 0) for fr in fold_results]),

        "hatnet_acc": np.mean([fr.get("hatnet_acc", 0) for fr in fold_results]),
        "hatnet_precision": np.mean([fr.get("hatnet_precision", 0) for fr in fold_results]),
        "hatnet_recall": np.mean([fr.get("hatnet_recall", 0) for fr in fold_results]),
        "hatnet_f1": np.mean([fr.get("hatnet_f1", 0) for fr in fold_results]),

        "TP": round(np.mean([fr.get("TP", 0) for fr in fold_results])),
        "FP": round(np.mean([fr.get("FP", 0) for fr in fold_results])),
        "TN": round(np.mean([fr.get("TN", 0) for fr in fold_results])),
        "FN": round(np.mean([fr.get("FN", 0) for fr in fold_results])),

        "TP_gsa": round(np.mean([fr.get("TP_gsa", 0) for fr in fold_results])),
        "FP_gsa": round(np.mean([fr.get("FP_gsa", 0) for fr in fold_results])),
        "TN_gsa": round(np.mean([fr.get("TN_gsa", 0) for fr in fold_results])),
        "FN_gsa": round(np.mean([fr.get("FN_gsa", 0) for fr in fold_results])),

        "TP_hatnet": round(np.mean([fr.get("TP_hatnet", 0) for fr in fold_results])),
        "FP_hatnet": round(np.mean([fr.get("FP_hatnet", 0) for fr in fold_results])),
        "TN_hatnet": round(np.mean([fr.get("TN_hatnet", 0) for fr in fold_results])),
        "FN_hatnet": round(np.mean([fr.get("FN_hatnet", 0) for fr in fold_results])),
    }

    ## Now you can use the previously provided code to plot the averaged confusion matrix
    # plot_confusion_matrix(avg_results['TP'], avg_results['FP'], avg_results['TN'], avg_results['FN'], 
    #                       title="DepDet Confusion Matrix", filename="mainkfold_depdet_confusion_matrix.png")
    # plot_confusion_matrix(avg_results['TP_gsa'], avg_results['FP_gsa'], avg_results['TN_gsa'], avg_results['FN_gsa'],
    #                       title="GSA Confusion Matrix", filename="mainkfold_gsa_confusion_matrix.png")
    # plot_confusion_matrix(avg_results['TP_hatnet'], avg_results['FP_hatnet'], avg_results['TN_hatnet'], avg_results['FN_hatnet'],
    #                       title="HatNet Confusion Matrix", filename="mainkfold_hatnet_confusion_matrix.png")


    ## Now you can use mean confusion matrix
    plot_confusion_matrix_mean(avg_results['TP'], avg_results['FP'], avg_results['TN'], avg_results['FN'], 
                          title="Mean DepDet Confusion Matrix", filename="mean_mainkfold_depdet_confusion_matrix.png")
    plot_confusion_matrix_mean(avg_results['TP_gsa'], avg_results['FP_gsa'], avg_results['TN_gsa'], avg_results['FN_gsa'],
                          title="Mean GSA Confusion Matrix", filename="mean_mainkfold_gsa_confusion_matrix.png")
    plot_confusion_matrix_mean(avg_results['TP_hatnet'], avg_results['FP_hatnet'], avg_results['TN_hatnet'], avg_results['FN_hatnet'],
                          title="Mean HatNet Confusion Matrix", filename="mean_mainkfold_hatnet_confusion_matrix.png")

    ##-----------------------------------------------------------------

    LOG_INFO(f"Average cross-validated results: {avg_results}", mcolor='green')

if __name__ == "__main__":
    main()


