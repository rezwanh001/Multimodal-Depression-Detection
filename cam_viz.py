# # import matplotlib.pyplot as plt
# # import numpy as np
# # import torch
# # import os 
# # import argparse
# # from tqdm import tqdm
# # import random
# # from datasets import get_dvlog_dataloader  # Assuming this is your data loader

# # -*- coding: utf-8 -*-
# '''
# @author: Md Rezwanul Haque
# '''
# #---------------------------------------------------------------
# # Imports
# #---------------------------------------------------------------
# import warnings
# warnings.filterwarnings('ignore')
# import os 
# import argparse
# import yaml
# from termcolor import colored
# import torch
# import numpy as np
# import random
# from tqdm import tqdm
# import torch.nn.functional as F
# from models import DepDet, DepressionDetector, DepressionDet
# from models import FocalLoss, CombinedLoss
# from datasets import get_dvlog_dataloader, kfold_get_dvlog_dataloader
# from utils import plot_confusion_matrix, plot_confusion_matrix_mean
# from torch.utils.data import ConcatDataset, Subset, DataLoader
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ##--------------------------------------

# # def extract_cam_weights(model, loader, device):
# #     """Extract CAM weights for all the samples in the given loader."""
# #     model.eval()
# #     all_weights = []

# #     with torch.no_grad():
# #         with tqdm(loader, desc="Extracting CAM Weights", leave=False, unit="batch") as pbar:
# #             for x, _ in pbar:  # We only need the input x here
# #                 x = x.to(device)
# #                 # Assuming get_cam_weights is a method of your model that extracts CAM weights
# #                 cam_weights = model.get_cam_weights(x)  # Shape should be (batch_size, time_steps)
# #                 print(f"cam_weights shape: {cam_weights.shape}")
# #                 all_weights.append(cam_weights.cpu().numpy())

# #     # Concatenate all batches to create a full matrix of CAM weights
# #     all_weights = np.concatenate(all_weights, axis=0)  # Shape (total_samples, time_steps)
# #     return all_weights

# # def extract_cam_weights(model, loader, device):
# #     """Extract CAM weights for all the samples in the given loader and keep them in variable-length format."""
# #     model.eval()
# #     all_weights = []

# #     with torch.no_grad():
# #         with tqdm(loader, desc="Extracting CAM Weights", leave=False, unit="batch") as pbar:
# #             for x, _ in pbar:  # We only need the input x here
# #                 x = x.to(device)
# #                 # Assuming get_cam_weights is a method of your model that extracts CAM weights
# #                 cam_weights = model.get_cam_weights(x)  # Shape (batch_size, time_steps)
# #                 print(f"cam_weights shape: {cam_weights.shape}")

# #                 # Split the batch into individual sequences and append each to all_weights
# #                 for i in range(cam_weights.shape[0]):
# #                     all_weights.append(cam_weights[i].cpu().numpy())  # Each item is (time_steps,)

# #     return all_weights


# # def extract_cam_weights(model, loader, device):
# #     """Extract CAM weights for all the samples in the given loader."""
# #     model.eval()
# #     all_weights = []

# #     min_length = float('inf')

# #     # First pass to find the minimum sequence length
# #     with torch.no_grad():
# #         for x, _ in loader:
# #             x = x.to(device)
# #             cam_weights = model.get_cam_weights(x)
# #             min_length = min(min_length, cam_weights.shape[1])  # Update min_length

# #     # Second pass to extract and truncate CAM weights
# #     with torch.no_grad():
# #         with tqdm(loader, desc="Extracting CAM Weights", leave=False, unit="batch") as pbar:
# #             for x, _ in pbar:
# #                 x = x.to(device)
# #                 cam_weights = model.get_cam_weights(x)  # Shape (batch_size, time_steps)

# #                 # Truncate cam_weights to min_length along dimension 1 (time dimension)
# #                 truncated_cam_weights = cam_weights[:, :min_length]

# #                 # Append to the list of all weights
# #                 all_weights.append(truncated_cam_weights.cpu().numpy())

# #     # Concatenate all batches to create a full matrix of CAM weights
# #     all_weights = np.concatenate(all_weights, axis=0)  # Shape (total_samples, min_length)
# #     return all_weights


# ###----------------------------------------------------------------------
# def extract_cam_weights(model, loader, device):
#     """Extract CAM weights for all the samples in the given loader."""
#     model.eval()
#     all_weights = []

#     max_length = 0

#     # First pass to find the maximum sequence length
#     with torch.no_grad():
#         for x, _ in loader:
#             x = x.to(device)
#             cam_weights = model.get_cam_weights(x)
#             max_length = max(max_length, cam_weights.shape[1])  # Update max_length
#             # print(f"max length: {max_length}")  

#     # Second pass to extract and pad CAM weights
#     with torch.no_grad():
#         with tqdm(loader, desc="Extracting CAM Weights", leave=False, unit="batch") as pbar:
#             for x, _ in pbar:
#                 x = x.to(device)
#                 cam_weights = model.get_cam_weights(x)  # Shape (batch_size, time_steps)
                
#                 # Pad cam_weights to max_length along dimension 1 (time dimension)
#                 padded_cam_weights = F.pad(cam_weights, (0, max_length - cam_weights.shape[1]), "constant", 0)

#                 # Append to the list of all weights
#                 all_weights.append(padded_cam_weights.cpu().numpy())

#     # Concatenate all batches to create a full matrix of CAM weights
#     all_weights = np.concatenate(all_weights, axis=0)  # Shape (total_samples, max_length)
#     return all_weights

# def plot_cam_weights(cam_weights, title="CAM Weight Visualization", aspect='auto'):
#     """Plot CAM weights as a heatmap."""
#     plt.figure(figsize=(12, 8))
#     plt.imshow(cam_weights, cmap='viridis', aspect=aspect) # type: ignore
#     plt.colorbar(label="CAM Weight")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Person ID")
#     plt.title(title)
#     plt.show()

# # import matplotlib.pyplot as plt
# # import numpy as np

# # def plot_cam_weights_variable_length(cam_weights_list, title="CAM Weight Visualization"):
# #     """
# #     Plot CAM weights as a heatmap-like plot with variable sequence lengths.
    
# #     Args:
# #         cam_weights_list (list of numpy arrays): List containing CAM weights of different lengths.
# #         title (str): The title of the plot.
# #     """
# #     # Determine the maximum sequence length
# #     max_length = max([len(weights) for weights in cam_weights_list])
    
# #     # Pad each sequence to the maximum length with NaN values
# #     padded_cam_weights = np.full((len(cam_weights_list), max_length), np.nan)  # Fill with NaNs initially
# #     for i, weights in enumerate(cam_weights_list):
# #         padded_cam_weights[i, :len(weights)] = weights  # Replace NaNs with actual values

# #     # Plot the padded CAM weights as a heatmap
# #     plt.figure(figsize=(12, 8))
# #     cmap = plt.cm.viridis
# #     cmap.set_bad(color='white')  # Set NaN values to white or any color of choice
# #     plt.imshow(padded_cam_weights, cmap=cmap, aspect='auto', interpolation='nearest')
# #     plt.colorbar(label="CAM Weight")
# #     plt.xlabel("Time Steps")
# #     plt.ylabel("Person ID")
# #     plt.title(title)
# #     plt.show()


# ##--------------------------------------


# def LOG_INFO(msg,mcolor='blue'):
#     '''
#         prints a msg/ logs an update
#         args:
#             msg     =   message to print
#             mcolor  =   color of the msg    
#     '''
#     print(colored("#LOG :",'green')+colored(msg,mcolor)) # type: ignore

# # Seed
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# set_seed(2024)

# CONFIG_PATH = "./main_config.yaml"

# def parse_args():
#     with open(CONFIG_PATH, "r") as f:
#         config = yaml.safe_load(f)

#     parser = argparse.ArgumentParser(
#         description="Train and test a model on the DVLOG dataset."
#     )
#     # arguments whose default values are in config.yaml
#     parser.add_argument("--data_dir", type=str)
#     parser.add_argument("--train_gender", type=str)
#     parser.add_argument("--test_gender", type=str)
#     parser.add_argument(
#         "-m", "--model", type=str,
#         choices=["DepressionDetector", "DepDet"]
#     )
#     parser.add_argument("-e", "--epochs", type=int)
#     parser.add_argument("-bs", "--batch_size", type=int)
#     parser.add_argument("-lr", "--learning_rate", type=float)
#     parser.add_argument(
#         "-sch", "--lr_scheduler", type=str,
#         choices=["cos", "None",]
#     )
#     parser.add_argument("-d", "--device", type=str, nargs="*")
#     parser.set_defaults(**config)
#     args = parser.parse_args()

#     return args


# def main():
#     args = parse_args()
#     LOG_INFO(args)

#     # construct the model
#     if args.model == "DepDet":
#         net = DepDet(d=256, l=6)
#     elif args.model == "DepressionDetector":
#         net = DepressionDetector(d=256, l=6, t_downsample=4)
#     elif args.model == "DepressionDet":
#         net = DepressionDet(d=256, l=6, t_downsample=4)

#     # Move model to the correct device
#     net = net.to(args.device[0])
#     if len(args.device) > 1:
#         net = torch.nn.DataParallel(net, device_ids=[int(i.split(':')[-1]) for i in args.device])

#     # Load the best saved model
#     net.load_state_dict(
#         torch.load(os.path.join("./weights", "new_best_model_main.pt"), map_location=args.device[0])
#     )

#     # Prepare the data
#     test_loader = get_dvlog_dataloader(
#         args.data_dir, "test", args.batch_size, args.test_gender
#     )

#     # Extract CAM weights from the test dataset
#     LOG_INFO("Extracting CAM weights for visualization...")
#     cam_weights = extract_cam_weights(net, test_loader, args.device[0])

#     # # Plot CAM weights for depressed people (assuming labels are available to split data)
#     # # Here we are simply plotting all weights, but you can also split by labels if needed.
#     plot_cam_weights(cam_weights, title="Visualization of CAM weight in depressed people")

#     # plot_cam_weights_variable_length(cam_weights, title="Visualization of CAM weight in depressed people")

# if __name__ == "__main__":
#     main()


##===========================================================================================

# import seaborn as sns
# import numpy as np
# import torch
# from models import DepDet
# from datasets import get_dvlog_dataloader
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Function to extract CAM weights
# def extract_cam_weights(model, device, data_loader):
#     cam_weights = []
#     labels = []

#     with torch.no_grad():
#         model.eval()
#         for x, y in data_loader:
#             x, y = x.to(device), y.to(device).unsqueeze(1)

#             # Extract features (assuming GSA predictions hold CAM weights)
#             z, gsa_pred, hatnet_pred = model.feature_extractor(x)

#             # Append CAM weights and labels
#             cam_weights.append(z.cpu().detach().numpy())  # Assuming gsa_pred is the CAM output
#             labels.append(y.cpu().detach().numpy())

#     # Concatenate along the first axis and ensure proper dimensions
#     cam_weights = np.concatenate(cam_weights)
#     labels = np.concatenate(labels).squeeze()  # Remove extra dimensions from labels if needed

#     return cam_weights, labels

# def plot_cam_heatmap(cam_weights, labels, class_list, save_path):
#     # Set figure size to increase the x-axis length
#     plt.figure(figsize=(10, 10))

#     # Separate CAM weights by class
#     for class_idx, class_name in enumerate(class_list):
#         plt.subplot(2, 1, class_idx + 1)

#         # Get the CAM weights for the specific class
#         class_cam_weights = cam_weights[labels == class_idx]
        
#         # Ensure that class_cam_weights has 2D shape (if needed, reshape)
#         if len(class_cam_weights.shape) == 1:
#             class_cam_weights = class_cam_weights.reshape(1, -1)  # Reshape to 2D

#         # Plot heatmap (transpose if needed to match axes like in the example image)
#         sns.heatmap(class_cam_weights, cmap="YlGnBu", cbar=True)

#         # Set the labels
#         plt.xlabel("sequence length", fontsize=12)
#         plt.ylabel("id number", fontsize=12)
        
#         # Add ticks for the x-axis and y-axis with labels
#         num_time_points = class_cam_weights.shape[1]
#         plt.xticks(ticks=[i for i in range(0, num_time_points, 50)], labels=[str(i) for i in range(0, num_time_points, 50)])
        
#         num_people = class_cam_weights.shape[0]
#         plt.yticks(ticks=[i for i in range(0, num_people, 25)], labels=[str(i) for i in range(0, num_people, 25)])

#         # # Title will be at the lower portion, using subtitle instead of main title
#         # plt.suptitle("")  # Clearing main title
#         # plt.figtext(0.5, 0.01, f"(a) Visualization of CAM weight in {class_name}", ha='center', fontsize=12)

#         # plt.title(f"(a) Visualization of CAM weight in {class_name}")

#     # # Adding titles at the bottom for both subplots
#     # plt.figtext(0.45, 0.48, "(a) Visualization of CAM weight in people with " + class_list[0], ha='center', fontsize=12)
#     # plt.figtext(0.45, 0.03, "(b) Visualization of CAM weight in people with " + class_list[1], ha='center', fontsize=12)


#     # Adding titles at the bottom for both subplots
#     plt.figtext(0.45, 0.48, "(a)", ha='center', fontsize=12)
#     plt.figtext(0.45, 0.03, "(b)", ha='center', fontsize=12)


#     # Adjust layout: add space between subplots using hspace
#     plt.subplots_adjust(hspace=0.7)  # Increase hspace for more vertical space between subplots

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add margin to fit the figure
#     plt.savefig(save_path, dpi=300)
#     plt.show()


# # Function to average CAM weights across folds
# def average_cam_weights_across_folds(cam_weights_list):
#     # Stack all CAM weights along a new axis (fold axis) and compute the mean across folds
#     cam_weights_stacked = np.stack(cam_weights_list, axis=0)
#     averaged_cam_weights = np.mean(cam_weights_stacked, axis=0)
#     return averaged_cam_weights

# # Main function for 10-fold cross-validation and averaging CAM weights
# def main():
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     # Load your data
#     data_dir = "./data/dvlog-dataset"
#     batch_size = 8
#     test_gender = "both"
#     train_gender = "both"
    
#     # Dataloader for test data
#     # dataloader
#     train_loader = get_dvlog_dataloader(data_dir, "train", batch_size, train_gender)
#     val_loader = get_dvlog_dataloader(data_dir, "valid", batch_size, test_gender)
#     test_loader = get_dvlog_dataloader(data_dir, "test", batch_size, test_gender)
    
#     class_list = ["depressed", "normal"]
    
#     # Placeholder for accumulating CAM weights from all folds
#     cam_weights_list = []
#     labels_list = []

#     # Loop through each fold (assuming the weights for each fold are saved separately)
#     for fold in range(10):
#         model = DepDet(d=256, l=6)

#         # Load the model weights for the current fold
#         model_path = f"/backup/backup/Rezwan/depression/weights/best_model_wo_lrs_{fold}.pt"
#         model.load_state_dict(torch.load(model_path))
#         model = model.to(device)

#         # Extract CAM weights from the test dataset
#         cam_weights, labels = extract_cam_weights(model, device, test_loader)

#         # Plot the CAM heatmaps for each class
#         print(f"Fold Number: {fold+1}")
#         plot_cam_heatmap(cam_weights, labels, class_list, f"/backup/backup/Rezwan/depression/results/TSNE/CAM_Heatmap_depression_{fold}.png")


#         # Accumulate CAM weights for averaging
#         cam_weights_list.append(cam_weights)

#         # Store labels (assuming labels are the same for all folds)
#         if fold == 0:
#             labels_list = labels  # Save only once, as labels should be the same

#     # Average CAM weights across all folds
#     averaged_cam_weights = average_cam_weights_across_folds(cam_weights_list)

#     # Plot the averaged CAM heatmaps for each class
#     print("Avg Plot!!!")
#     plot_cam_heatmap(averaged_cam_weights, labels_list, class_list, "/backup/backup/Rezwan/depression/results/TSNE/CAM_Heatmap_depression_avg.png")

# if __name__ == "__main__":
#     main()

##========================================================================
# from models import DepDet

# def average_model_weights(model_class, model_paths, device):
#     # Load the first model as a template
#     averaged_model = model_class(d=256, l=6)
#     averaged_model.load_state_dict(torch.load(model_paths[0], map_location=device))
#     averaged_model.to(device)

#     # Accumulate weights
#     state_dicts = [torch.load(path, map_location=device) for path in model_paths]
#     averaged_state_dict = averaged_model.state_dict()

#     for key in averaged_state_dict:
#         # Average weights across all models
#         averaged_state_dict[key] = torch.mean(torch.stack([state_dict[key] for state_dict in state_dicts]), dim=0)

#     # Load averaged weights back into the model
#     averaged_model.load_state_dict(averaged_state_dict)
#     return averaged_model

# # Example usage
# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_paths = [f"./weights/best_model_wo_lrs_{i}.pt" for i in range(10)]
# averaged_model = average_model_weights(DepDet, model_paths, device)
##========================================================================

# import torch
# from models import DepDet

# def average_model_weights(model_class, model_paths, device):
#     # Load the first model as a template
#     averaged_model = model_class(d=256, l=6)
#     averaged_model.load_state_dict(torch.load(model_paths[0], map_location=device))
#     averaged_model.to(device)

#     # Accumulate weights
#     state_dicts = [torch.load(path, map_location=device) for path in model_paths]
#     averaged_state_dict = averaged_model.state_dict()

#     for key in averaged_state_dict:
#         # Average weights across all models
#         averaged_state_dict[key] = torch.mean(torch.stack([state_dict[key] for state_dict in state_dicts]), dim=0)

#     # Load averaged weights back into the model
#     averaged_model.load_state_dict(averaged_state_dict)
    
#     return averaged_model

# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_paths = [f"./weights/best_model_wo_lrs_{i}.pt" for i in range(10)]
# averaged_model = average_model_weights(DepDet, model_paths, device)

# # Save the averaged model weights
# save_path = "./weights/best_averaged_model_weights.pt"
# torch.save(averaged_model.state_dict(), save_path)
# print(f"Averaged model weights saved to {save_path}")

####-------------------------------------------------------------------------------------------------------------------------

###==============================All weights Cam Viz from 10-folds================================
import seaborn as sns
import numpy as np
import torch
from models import DepDet
from datasets import get_dvlog_dataloader
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Function to extract CAM weights
def extract_cam_weights(model, device, data_loader):
    cam_weights = []
    labels = []

    with torch.no_grad():
        model.eval()
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            # Extract features (assuming GSA predictions hold CAM weights)
            z, gsa_pred, hatnet_pred = model.feature_extractor(x)

            # Append CAM weights and labels
            cam_weights.append(z.cpu().detach().numpy())  # Assuming gsa_pred is the CAM output
            labels.append(y.cpu().detach().numpy())

    # Concatenate along the first axis and ensure proper dimensions
    cam_weights = np.concatenate(cam_weights)
    labels = np.concatenate(labels).squeeze()  # Remove extra dimensions from labels if needed

    return cam_weights, labels

def plot_cam_heatmap(cam_weights, labels, class_list, save_path):
    # Set figure size to increase the x-axis length
    plt.figure(figsize=(10, 10))

    # Separate CAM weights by class
    for class_idx, class_name in enumerate(class_list):
        plt.subplot(2, 1, class_idx + 1)

        # Get the CAM weights for the specific class
        class_cam_weights = cam_weights[labels == class_idx]
        
        # Ensure that class_cam_weights has 2D shape (if needed, reshape)
        if len(class_cam_weights.shape) == 1:
            class_cam_weights = class_cam_weights.reshape(1, -1)  # Reshape to 2D

        # Plot heatmap (transpose if needed to match axes like in the example image)
        sns.heatmap(class_cam_weights, cmap="YlGnBu", cbar=True)

        # Set the labels
        plt.xlabel("sequence length", fontweight='bold')
        plt.ylabel("id number", fontweight='bold')
        
        # Add ticks for the x-axis and y-axis with labels
        num_time_points = class_cam_weights.shape[1]
        plt.xticks(ticks=[i for i in range(0, num_time_points, 50)], labels=[str(i) for i in range(0, num_time_points, 50)])
        
        num_people = class_cam_weights.shape[0]
        plt.yticks(ticks=[i for i in range(0, num_people, 25)], labels=[str(i) for i in range(0, num_people, 25)])

        # # Title will be at the lower portion, using subtitle instead of main title
        # plt.suptitle("")  # Clearing main title
        # plt.figtext(0.5, 0.01, f"(a) Visualization of CAM weight in {class_name}", ha='center', fontsize=12)

        # plt.title(f"(a) Visualization of CAM weight in {class_name}")

    # # Adding titles at the bottom for both subplots
    # plt.figtext(0.45, 0.48, "(a) Visualization of CAM weight in people with " + class_list[0], ha='center', fontsize=12)
    # plt.figtext(0.45, 0.03, "(b) Visualization of CAM weight in people with " + class_list[1], ha='center', fontsize=12)


    # Adding titles at the bottom for both subplots
    plt.figtext(0.45, 0.48, "(a)", ha='center', fontweight=16)
    plt.figtext(0.45, 0.03, "(b)", ha='center', fontweight=16)


    # Adjust layout: add space between subplots using hspace
    plt.subplots_adjust(hspace=0.8)  # Increase hspace for more vertical space between subplots

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore # Add margin to fit the figure
    plt.savefig(save_path, dpi=300)
    # plt.show()


# Function to average CAM weights across folds
def average_cam_weights_across_folds(cam_weights_list):
    # Stack all CAM weights along a new axis (fold axis) and compute the mean across folds
    cam_weights_stacked = np.stack(cam_weights_list, axis=0)
    averaged_cam_weights = np.mean(cam_weights_stacked, axis=0)
    return averaged_cam_weights

# Main function for 10-fold cross-validation and averaging CAM weights
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load your data
    data_dir = "./data/dvlog-dataset"
    batch_size = 8
    test_gender = "both"
    train_gender = "both"
    
    # Dataloader for test data
    # dataloader
    train_loader = get_dvlog_dataloader(data_dir, "train", batch_size, train_gender)
    val_loader = get_dvlog_dataloader(data_dir, "valid", batch_size, test_gender)
    test_loader = get_dvlog_dataloader(data_dir, "test", batch_size, test_gender)
    
    # class_list = ["depressed", "normal"]

    class_list = ["normal", "depression"] #### this is classification list
    
    # Placeholder for accumulating CAM weights from all folds
    cam_weights_list = []
    labels_list = []

    # Loop through each fold (assuming the weights for each fold are saved separately)
    for fold in range(10):
        model = DepDet(d=256, l=6)

        # Load the model weights for the current fold
        model_path = f"./weights/best_model_wo_lrs_{fold}.pt"
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        # Extract CAM weights from the test dataset
        cam_weights, labels = extract_cam_weights(model, device, test_loader)

        # Plot the CAM heatmaps for each class
        print(f"Fold Number: {fold+1}")
        plot_cam_heatmap(cam_weights, labels, class_list, f"./results/TSNE/CAM_Heatmap_depression_{fold+1}.png")

        # Accumulate CAM weights for averaging
        cam_weights_list.append(cam_weights)

        # Store labels (assuming labels are the same for all folds)
        if fold == 0:
            labels_list = labels  # Save only once, as labels should be the same

    # Average CAM weights across all folds
    averaged_cam_weights = average_cam_weights_across_folds(cam_weights_list)

    # Plot the averaged CAM heatmaps for each class
    print("Avg Plot!!!")
    plot_cam_heatmap(averaged_cam_weights, labels_list, class_list, "./results/TSNE/CAM_Heatmap_depression_avg.png")

if __name__ == "__main__":
    main()
###=========================================================================================

####-------------------------------------------------------------------------------------------------------------------------

###==============================Avaraging weights from 10-folds================================
# import torch
# from models import DepDet

# def average_model_weights(model_class, model_paths, device):
#     # Load the first model as a template
#     averaged_model = model_class(d=256, l=6)
#     averaged_model.load_state_dict(torch.load(model_paths[0], map_location=device))
#     averaged_model.to(device)

#     # Accumulate weights
#     state_dicts = [torch.load(path, map_location=device) for path in model_paths]
#     averaged_state_dict = averaged_model.state_dict()

#     for key in averaged_state_dict:
#         # Stack the weights across models, converting them to float to avoid errors
#         stacked_weights = torch.stack([state_dict[key].float() for state_dict in state_dicts])
#         # Average the weights across all models
#         averaged_state_dict[key] = torch.mean(stacked_weights, dim=0)

#     # Load averaged weights back into the model
#     averaged_model.load_state_dict(averaged_state_dict)
#     return averaged_model

# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_paths = [f"./weights/best_model_wo_lrs_{i}.pt" for i in range(10)]
# averaged_model = average_model_weights(DepDet, model_paths, device)

# # Save the averaged model weights
# save_path = "./weights/averaged_model_weights.pt"
# torch.save(averaged_model.state_dict(), save_path)
# print(f"Averaged model weights saved to {save_path}")
###========================================================================

####-------------------------------------------------------------------------------------------------------------------------

# import seaborn as sns
# import numpy as np
# import torch
# from models import DepDet
# from datasets import get_dvlog_dataloader
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Function to extract CAM weights
# def extract_cam_weights(model, device, data_loader):
#     cam_weights = []
#     labels = []

#     with torch.no_grad():
#         model.eval()
#         for x, y in data_loader:
#             x, y = x.to(device), y.to(device).unsqueeze(1)

#             # Extract features (assuming GSA predictions hold CAM weights)
#             z, gsa_pred, hatnet_pred = model.feature_extractor(x)

#             # Append CAM weights and labels
#             cam_weights.append(z.cpu().detach().numpy())  # Assuming gsa_pred is the CAM output
#             labels.append(y.cpu().detach().numpy())

#     # Concatenate along the first axis and ensure proper dimensions
#     cam_weights = np.concatenate(cam_weights)
#     labels = np.concatenate(labels).squeeze()  # Remove extra dimensions from labels if needed

#     return cam_weights, labels

# def plot_cam_heatmap(cam_weights, labels, class_list, save_path):
#     # Set figure size to increase the x-axis length
#     plt.figure(figsize=(10, 10))

#     # Separate CAM weights by class
#     for class_idx, class_name in enumerate(class_list):
#         plt.subplot(2, 1, class_idx + 1)

#         # Get the CAM weights for the specific class
#         class_cam_weights = cam_weights[labels == class_idx]
        
#         # Ensure that class_cam_weights has 2D shape (if needed, reshape)
#         if len(class_cam_weights.shape) == 1:
#             class_cam_weights = class_cam_weights.reshape(1, -1)  # Reshape to 2D

#         # Plot heatmap (transpose if needed to match axes like in the example image)
#         sns.heatmap(class_cam_weights, cmap="YlGnBu", cbar=True)

#         # Set the labels
#         plt.xlabel("sequence length", fontweight='bold')
#         plt.ylabel("id number", fontweight='bold')
        
#         # Add ticks for the x-axis and y-axis with labels
#         num_time_points = class_cam_weights.shape[1]
#         plt.xticks(ticks=[i for i in range(0, num_time_points, 50)], labels=[str(i) for i in range(0, num_time_points, 50)])
        
#         num_people = class_cam_weights.shape[0]
#         plt.yticks(ticks=[i for i in range(0, num_people, 25)], labels=[str(i) for i in range(0, num_people, 25)])

#         # # Title will be at the lower portion, using subtitle instead of main title
#         # plt.suptitle("")  # Clearing main title
#         # plt.figtext(0.5, 0.01, f"(a) Visualization of CAM weight in {class_name}", ha='center', fontsize=12)

#         # plt.title(f"(a) Visualization of CAM weight in {class_name}")

#     # # Adding titles at the bottom for both subplots
#     # plt.figtext(0.45, 0.48, "(a) Visualization of CAM weight in people with " + class_list[0], ha='center', fontsize=12)
#     # plt.figtext(0.45, 0.03, "(b) Visualization of CAM weight in people with " + class_list[1], ha='center', fontsize=12)


#     # Adding titles at the bottom for both subplots
#     plt.figtext(0.45, 0.48, "(a)", ha='center', fontweight=16)
#     plt.figtext(0.45, 0.03, "(b)", ha='center', fontweight=16)


#     # Adjust layout: add space between subplots using hspace
#     plt.subplots_adjust(hspace=0.8)  # Increase hspace for more vertical space between subplots

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore # Add margin to fit the figure
#     plt.savefig(save_path, dpi=300)
#     plt.show()


# # Function to average CAM weights across folds
# def average_cam_weights_across_folds(cam_weights_list):
#     # Stack all CAM weights along a new axis (fold axis) and compute the mean across folds
#     cam_weights_stacked = np.stack(cam_weights_list, axis=0)
#     averaged_cam_weights = np.mean(cam_weights_stacked, axis=0)
#     return averaged_cam_weights

# # Main function for 10-fold cross-validation and averaging CAM weights
# def main():
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     # Load your data
#     data_dir = "./data/dvlog-dataset"
#     batch_size = 8
#     test_gender = "both"
#     train_gender = "both"
    
#     # Dataloader for test data
#     # dataloader
#     train_loader = get_dvlog_dataloader(data_dir, "train", batch_size, train_gender)
#     val_loader = get_dvlog_dataloader(data_dir, "valid", batch_size, test_gender)
#     test_loader = get_dvlog_dataloader(data_dir, "test", batch_size, test_gender)
    
#     # class_list = ["depressed", "normal"]

#     ## class names
#     class_list = ["normal", "depression"]

#     #model
#     model = DepDet(d=256, l=6)
    
#     # Load the model weights for the current fold
#     model_path = f"./weights/averaged_model_weights.pt"
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)

#     # Extract CAM weights from the test dataset
#     cam_weights, labels = extract_cam_weights(model, device, test_loader)

#     # Plot the averaged CAM heatmaps for each class
#     print("Avg Plot!!!")
#     plot_cam_heatmap(cam_weights, labels, class_list, "./results/TSNE/CAM_Heatmap_depression_avg.png")

# if __name__ == "__main__":
#     main()

###========================================================================

####-------------------------------------------------------------------------------------------------------------------------