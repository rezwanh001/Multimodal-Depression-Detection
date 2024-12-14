
# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

def plot_confusion_matrix(TP, FP, TN, FN, title="Confusion Matrix", filename="confusion_matrix.png"):
    """
    Plots and saves a confusion matrix using the provided values for TP, FP, TN, FN.

    Args:
        TP (int): True Positives
        FP (int): False Positives
        TN (int): True Negatives
        FN (int): False Negatives
        title (str): The title for the plot (default is "Confusion Matrix")
        filename (str): The name of the file to save the plot as (default is "confusion_matrix.png")
    """
    res_dir = "./results/conf_mat/"
    cm = [[TN, FP],
          [FN, TP]]  # Confusion matrix in the form [[TN, FP], [FN, TP]]

    # # Create a heatmap
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    #             xticklabels=["normal", "depression"],
    #             yticklabels=["normal", "depression"])

    # # plt.title(title)
    # plt.ylabel('True Label', fontweight='bold')
    # plt.xlabel('Predicted Label', fontweight='bold')

    # Create the heatmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["normal", "depression"],
                    yticklabels=["normal", "depression"])

    # Set the x-axis tick labels to bold
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')

    # Set the y-axis tick labels to bold (optional)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    # Set labels for axes
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(res_dir, filename), dpi=300)
    plt.close()  # Close the plot to avoid displaying it when not necessary

def plot_confusion_matrix_mean(TP, FP, TN, FN, title="Confusion Matrix", filename="confusion_matrix.png"):
    """
    Plots and saves a confusion matrix using the provided values for TP, FP, TN, FN.

    Args:
        TP (int): True Positives
        FP (int): False Positives
        TN (int): True Negatives
        FN (int): False Negatives
        title (str): The title for the plot (default is "Confusion Matrix")
        filename (str): The name of the file to save the plot as (default is "confusion_matrix.png")
    """
    res_dir = "./results/conf_mat/"
    cm = [[TN, FP],
          [FN, TP]]  # Confusion matrix in the form [[TN, FP], [FN, TP]]

    cm = np.array(cm).astype('float') / np.array(cm).sum(axis=1)[:, np.newaxis]

    # # Create a heatmap
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', cbar=False,
    #             xticklabels=["normal", "depression"],
    #             yticklabels=["normal", "depression"])

    # # plt.title(title)
    # plt.ylabel('True Label', fontweight='bold')
    # plt.xlabel('Predicted Label', fontweight='bold')

    # Create the heatmap
    ax = sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', cbar=False,
                    xticklabels=["normal", "depression"],
                    yticklabels=["normal", "depression"])

    # Set the x-axis tick labels to bold
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')

    # Set the y-axis tick labels to bold (optional)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    # Set labels for axes
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(res_dir, filename), dpi=300)
    plt.close()  # Close the plot to avoid displaying it when not necessary


def plot_tsne(features, labels, title="t-SNE plot", filename="t_sne.png"):

    ## save the png file
    res_dir = "./results/TSNE/"

    ## class names
    class_list = ["normal", "depression"]

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_embedded = tsne.fit_transform(features)

    # Plot the t-SNE embeddings
    plt.rcParams['axes.linewidth'] = 1.5 #set the value globally
    plt.figure(figsize=(8, 6))
    for i in tqdm(range(len(np.unique(labels)))):
        indices = np.where(labels == i)
        plt.scatter(features_embedded[indices, 0], features_embedded[indices, 1], label=f'{class_list[i]}', s=10)

    # Set the limits for x-axis and y-axis
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    
    # Set specific tick values for x-axis and y-axis
    plt.xticks(np.arange(-25, 26, 5))  # Adjust the range and step as needed
    plt.yticks(np.arange(-25, 26, 5))  # Adjust the range and step as needed

    # plt.title('t-SNE Visualization of Features')
    plt.xlabel('Dimension 1', fontweight='bold')
    plt.ylabel('Dimension 2', fontweight='bold')
    
    # plt.legend()
    # Place the legend in the top right corner
    plt.legend(loc='upper right')

    # plt.savefig(f"./viz_BioVid/res_images/TSNE_Curve_{file_name}_BioVid.png", dpi=300)
    # plt.show()

    # Save the plot as a PNG file
    plt.savefig(os.path.join(res_dir, filename), dpi=300)
    plt.close()  # Close the plot to avoid displaying it when not necessary

def improve_tsne_visualization(features, labels, use_umap=False, tsne_perplexity=30, tsne_learning_rate=100, umap_neighbors=15, umap_min_dist=0.1, filename="improve_tsne.png"):
    """
    Improves the visualization of high-dimensional features using t-SNE or UMAP.
    This function aims to improve the quality of t-SNE visualization for features extracted from a neural network model.
    It includes normalization of features, hyperparameter tuning for t-SNE, and also provides an alternative UMAP for dimensionality reduction.

    
    Args:
        features (numpy.ndarray): Feature matrix of shape (num_samples, num_features).
        labels (numpy.ndarray): Corresponding labels of shape (num_samples,).
        use_umap (bool): If True, use UMAP instead of t-SNE. Default is False.
        tsne_perplexity (int): Perplexity parameter for t-SNE. Default is 30.
        tsne_learning_rate (float): Learning rate parameter for t-SNE. Default is 100.
        umap_neighbors (int): Number of neighbors for UMAP. Default is 15.
        umap_min_dist (float): Minimum distance for UMAP. Default is 0.1.
        
    Returns:
        None: The function will display the t-SNE or UMAP plot of the features.
    """
    ## save the png file
    res_dir = "./results/TSNE/"

    ## class names
    class_list = ["normal", "depression"]


    # Step 1: Normalize features
    print("Normalizing the features...")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Step 2: Perform dimensionality reduction (either t-SNE or UMAP)
    if use_umap:
        # Using UMAP for dimensionality reduction
        print("Applying UMAP for dimensionality reduction...")
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=42) # type: ignore
        reduced_features = reducer.fit_transform(normalized_features)
    else:
        # Using t-SNE for dimensionality reduction
        print(f"Applying t-SNE with perplexity={tsne_perplexity} and learning_rate={tsne_learning_rate}...")
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate=tsne_learning_rate, random_state=42)
        reduced_features = tsne.fit_transform(normalized_features)

    # Step 3: Plot the results
    print("Plotting the dimensionality reduction results...")

    # plt.figure(figsize=(10, 8))
    # unique_labels = np.unique(labels)
    # for label in unique_labels:
    #     indices = labels == label
    #     plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label="normal" if label == 0 else "depression", s=10) # type: ignore

    # Plot the t-SNE embeddings
    plt.rcParams['axes.linewidth'] = 1.5 #set the value globally
    plt.figure(figsize=(8, 6))
    for i in tqdm(range(len(np.unique(labels)))):
        indices = np.where(labels == i)
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'{class_list[i]}', s=10)  # type: ignore

    # Set the limits for x-axis and y-axis
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    
    # Set specific tick values for x-axis and y-axis
    plt.xticks(np.arange(-40, 41, 10))  # Adjust the range and step as needed
    plt.yticks(np.arange(-40, 41, 10))  # Adjust the range and step as needed

    # plt.title('t-SNE Visualization of Features')
    plt.xlabel('Dimension 1', fontweight='bold')
    plt.ylabel('Dimension 2', fontweight='bold')

    # plt.legend()
    # title = "UMAP Plot of Features" if use_umap else "t-SNE Plot of Features"
    # plt.title(title, fontsize=15)
    # # plt.show()

    # plt.legend()
    # Place the legend in the top right corner
    plt.legend(loc='upper right')

    # Save the plot as a PNG file
    plt.savefig(os.path.join(res_dir, filename), dpi=300)
    plt.close()  # Close the plot to avoid displaying it when not necessary

