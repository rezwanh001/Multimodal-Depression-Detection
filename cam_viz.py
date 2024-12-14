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


    # # Adding titles at the bottom for both subplots
    # plt.figtext(0.45, 0.48, "(a)", ha='center', fontweight=16)
    # plt.figtext(0.45, 0.03, "(b)", ha='center', fontweight=16)

    # Adding titles at the bottom for both subplots
    plt.figtext(0.45, 0.48, "(i)", ha='center', fontweight=16)
    plt.figtext(0.45, 0.03, "(ii)", ha='center', fontweight=16)


    # Adjust layout: add space between subplots using hspace
    plt.subplots_adjust(hspace=0.8)  # Increase hspace for more vertical space between subplots

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore # Add margin to fit the figure
    plt.savefig(save_path, dpi=300)
    plt.show()


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
    
    class_list = ["depressed", "normal"]
    
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
