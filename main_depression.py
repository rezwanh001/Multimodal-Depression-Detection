# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import argparse
import yaml
import wandb
import torch
import numpy as np
import random
from tqdm import tqdm
from models import DepDet, DepressionDetector
from models import FocalLoss
from datasets import get_dvlog_dataloader

# Seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

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

def train_epoch(
    net, train_loader, loss_fn, optimizer, lr_scheduler, device, 
    current_epoch, total_epochs
):
    """One training epoch."""
    net.train()
    sample_count = 0
    running_loss = 0.
    # running_gsa_loss = 0.
    # running_hatnet_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False, unit="batch"
    ) as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            y_pred = net(x)
            ## pass to Model
            # y_pred, gsa_pred, hatnet_pred = net(x)
            # gsa_pred, y_pred, hatnet_pred = net(x)
            # gsa_pred, hatnet_pred, y_pred  = net(x) 

            ## depdet loss 
            loss = loss_fn(y_pred, y.to(torch.float32))

            # ## gsa loss
            # loss_gsa = loss_fn(gsa_pred, y.to(torch.float32))

            # ## hatnet loss  
            # loss_hatnet = loss_fn(hatnet_pred, y.to(torch.float32))

            # total_loss = loss + loss_gsa + loss_hatnet

            # total_loss.backward()
            loss.backward()
            # loss_gsa.backward()
            # loss_hatnet.backward()

            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]

            # running_gsa_loss += loss_gsa.item() * x.shape[0]
            # running_hatnet_loss += loss_hatnet.item() * x.shape[0]

            # binary classification with only one output neuron
            pred = (y_pred > 0).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                # "loss_gsa": running_gsa_loss / sample_count,
                # "loss_hatnet": running_hatnet_loss / sample_count,
                "acc": correct_count / sample_count,
            })

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "loss": running_loss / sample_count,
        # "loss_gsa": running_gsa_loss / sample_count,
        # "loss_hatnet": running_hatnet_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(
    net, val_loader, loss_fn, device, 
):
    """Test the model on the validation / test set."""
    net.eval()
    sample_count = 0
    running_loss = 0.
    # running_gsa_loss = 0.
    # running_hatnet_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch"
        ) as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device).unsqueeze(1)

                y_pred = net(x)

                ## Pass to Model
                # y_pred, gsa_pred, hatnet_pred = net(x)
                # gsa_pred, y_pred, hatnet_pred = net(x) 
                # gsa_pred, hatnet_pred, y_pred  = net(x) 

                ## depdet loss (total loss)
                loss = loss_fn(y_pred, y.to(torch.float32))

                # ## gsa loss
                # loss_gsa = loss_fn(gsa_pred, y.to(torch.float32))

                # ## hatnet loss  
                # loss_hatnet = loss_fn(hatnet_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]

                # running_gsa_loss += loss_gsa.item() * x.shape[0]
                # running_hatnet_loss += loss_hatnet.item() * x.shape[0]

                # binary classification with only one output neuron
                pred = (y_pred > 0).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
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

                pbar.set_postfix({
                    "loss": l, 
                    # "loss_gsa": running_gsa_loss / sample_count,
                    # "loss_hatnet": running_hatnet_loss / sample_count,
                    "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
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
    return {
        "loss": l, 
        # "loss_gsa": running_gsa_loss / sample_count,
        # "loss_hatnet": running_hatnet_loss / sample_count,
        "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }


def main():
    args = parse_args()

    # initialize wandb
    wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
    wandb.init(
        project="dvlog", entity="rezwanh001", config=args, name=wandb_run_name, # type: ignore
    )
    args = wandb.config
    print(args)

    # construct the model
    if args.model == "DepDet":
        net = DepDet(d=256, l=6) 
    elif args.model == "DepressionDetector":
        net = DepressionDetector(d=256, l=6, t_downsample=4)

    net = net.to(args.device[0])
    if len(args.device) > 1:
        net = torch.nn.DataParallel(net, device_ids=args.device)

    # prepare the data
    train_loader = get_dvlog_dataloader(
        args.data_dir, "train", args.batch_size, args.train_gender
    )
    val_loader = get_dvlog_dataloader(
        args.data_dir, "valid", args.batch_size, args.test_gender
    )

    test_loader = get_dvlog_dataloader(
        args.data_dir, "test", args.batch_size, args.test_gender
    )

    # # set other training components
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, alpha=0.99, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # loss_fn = torch.nn.SmoothL1Loss()
    # optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.lr_scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20 
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        lr_scheduler = None

    # # set other training components
    # loss_fn = FocalLoss(alpha=1, gamma=2)  # Replace with custom loss function
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # if args.lr_scheduler == "cos":
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 10 
    #     )
    # else:
    #     lr_scheduler = None

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        train_results = train_epoch(
            net, train_loader, loss_fn, optimizer, lr_scheduler, 
            args.device[0], epoch, args.epochs
        )
        val_results = val(net, val_loader, loss_fn, args.device[0])

        val_acc = val_results["acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save the best model
            torch.save(net.state_dict(), f"{wandb.run.dir}/best_model.pt") # type: ignore
        
        wandb.log({
            "loss/train_loss": train_results["loss"],
            "acc/train_acc": train_results["acc"],
            "loss/val_loss": val_results["loss"],
            "acc/val_acc": val_results["acc"],
            "precision/val_precision": val_results["precision"],
            "recall/val_recall": val_results["recall"],
            "f1/val_f1": val_results["f1"],
        })

    wandb.run.summary["acc/best_val_acc"] = best_val_acc # type: ignore

    # upload the best model to wandb website
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f"{wandb.run.dir}/best_model.pt") # type: ignore
    wandb.log_artifact(artifact)

    # load the best model for testing
    net.load_state_dict(
        torch.load(f"{wandb.run.dir}/best_model.pt", map_location=args.device[0]) # type: ignore
    )
    test_results = val(net, test_loader, loss_fn, args.device[0])
    print("Test results:")
    print(test_results)
    wandb.run.summary["acc/test_acc"] = test_results["acc"] # type: ignore
    wandb.run.summary["loss/test_loss"] = test_results["loss"] # type: ignore
    wandb.run.summary["precision/test_precision"] = test_results["precision"] # type: ignore
    wandb.run.summary["recall/test_recall"] = test_results["recall"] # type: ignore
    wandb.run.summary["f1/test_f1"] = test_results["f1"] # type: ignore

    wandb.finish()

if __name__ == "__main__":
    main()
