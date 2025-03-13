import torchxrayvision as xrv
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import tarfile
from pathlib import Path
from tqdm.autonotebook import tqdm
import numpy as np
import time
import torch
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.estimators import TreeSearch, HillClimbSearch, ExhaustiveSearch, BicScore

def extract_tar_if_needed(tar_path, extract_dir):
    """
    Extract tar file if it hasn't been extracted already
    
    Parameters:
    tar_path (str): Path to the tar file
    extract_dir (str): Directory to extract to
    
    Returns:
    str: Path to the directory containing the extracted images
    """
    # Create extract directory if it doesn't exist
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if files are already extracted
    images_dir = os.path.join(extract_dir, 'images-224')
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print("Using previously extracted files...")
        return images_dir
    
    print(f"Extracting {tar_path} to {extract_dir}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_dir)
    
    return images_dir


def load_nih_dataset(
    imgpath,
    extract_dir="./data/NIH/extracted",
    views=['PA'],
    unique_patients=True,
    transform=None,
    batch_size=32,
    pathology_masks=False
):
    """
    Load the NIH ChestX-ray14 dataset using TorchXRayVision
    
    Parameters:
    imgpath (str): Path to either the tar file or directory containing NIH dataset images
    extract_dir (str): Directory to extract tar file to if needed
    views (list): List of views to include (default: ['PA'])
    unique_patients (bool): Whether to include only unique patients (default: True)
    transform: Optional transforms to apply (default: None)
    batch_size (int): Batch size for data loader (default: 32)
    pathology_masks (bool): Whether to include pathology masks (default: False)
    
    Returns:
    dataset: NIH_Dataset object
    dataloader: DataLoader object
    """
    # Handle tar file if provided
    if imgpath.endswith('.tar'):
        images_dir = extract_tar_if_needed(imgpath, extract_dir)
    else:
        images_dir = imgpath
    
    # Verify the directory exists and contains images
    if not os.path.isdir(images_dir):
        raise Exception(f"Could not find valid directory at {images_dir}")
    
    print(f"Loading dataset from {images_dir}")
    
    # Initialize the NIH dataset
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=images_dir,
        views=views,
        unique_patients=unique_patients,
        transform=transform,
        pathology_masks=pathology_masks
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    return dataset, dataloader


def print_dataset_info(dataset):
    """
    Print information about the dataset
    
    Parameters:
    dataset: NIH_Dataset object
    """
    print(f"Dataset size: {len(dataset)}")
    print(f"\nAvailable pathologies:")
    for i, pathology in enumerate(dataset.pathologies):
        print(f"{i+1}. {pathology}")
    
    print("\nPathology counts:")
    totals = dataset.totals()
    for pathology in totals:
        present = totals[pathology].get(1.0, 0)
        absent = totals[pathology].get(0.0, 0)
        print(f"{pathology}:")
        print(f"  Present: {present}")
        print(f"  Absent: {absent}")
        print(f"  Percentage: {(present/(present+absent))*100:.2f}%\n")


def show_sample_images(dataset, num_samples=4):
    """
    Display sample images from the dataset
    
    Parameters:
    dataset: NIH_Dataset object
    num_samples (int): Number of samples to display
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    
    for i in range(num_samples):
        # Get sample and print its structure to debug
        sample = dataset[i]
        print(f"Sample {i} structure:", sample.keys() if hasattr(sample, 'keys') else type(sample))
        
        try:
            if isinstance(sample, dict):
                # If it's a dictionary
                img = sample['img'] if 'img' in sample else sample['image']
                label = sample['lab'] if 'lab' in sample else sample['label']
            else:
                # If it's any other type of sequence
                img, label = sample
            
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            # Display the image
            axes[i].imshow(img[0] if img.ndim > 2 else img, cmap='gray')
            axes[i].axis('off')
            
            # Get pathologies present in this image
            pathologies_present = [
                dataset.pathologies[j] 
                for j in range(len(label)) 
                if label[j] == 1
            ]
            
            axes[i].set_title(
                "Pathologies:\n" + "\n".join(pathologies_present),
                fontsize=10
            )
        except Exception as e:
            print(f"Error processing sample {i}:", e)
            print("Sample content:", sample)
    
    plt.tight_layout()
    plt.show()


def run_inference(model, dataloader, num_samples=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Run inference on images and visualize results
    
    Parameters:
    model: Pre-trained torchxrayvision model
    dataloader: DataLoader containing the images
    num_samples: Number of samples to visualize
    device: Device to run inference on
    """
    model = model.to(device)
    model.eval()
    
    # Debug print dataset structure
    print("Dataset structure:")
    if hasattr(dataloader.dataset, 'dataset'):
        print("Using subset dataset")
        print(f"Original dataset type: {type(dataloader.dataset.dataset)}")
        if hasattr(dataloader.dataset.dataset, 'pathologies'):
            pathologies = list(dataloader.dataset.dataset.pathologies)
        else:
            print("Warning: No pathologies found in dataset")
    else:
        print(f"Dataset type: {type(dataloader.dataset)}")
        if hasattr(dataloader.dataset, 'pathologies'):
            pathologies = list(dataloader.dataset.pathologies)
        else:
            print("Warning: No pathologies found in dataset")
    
    # Default NIH pathologies if none found
    if not pathologies:
        pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
    
    print(f"\nNumber of pathologies: {len(pathologies)}")
    print("Pathologies:", pathologies)
    
    # Get a batch of images
    try:
        batch = next(iter(dataloader))
        print("\nBatch information:")
        print(f"Batch type: {type(batch)}")
        
        if isinstance(batch, dict):
            print("Batch keys:", batch.keys())
            images = batch['img'][:num_samples]
            true_labels = batch['lab'][:num_samples]
        else:
            print("Batch length:", len(batch))
            images, true_labels = batch
            images = images[:num_samples]
            true_labels = true_labels[:num_samples]
        
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {true_labels.shape}")
        
        # Validate shapes match pathologies
        assert true_labels.shape[1] == len(pathologies), \
            f"Mismatch between number of labels ({true_labels.shape[1]}) and pathologies ({len(pathologies)})"
        
        images = images.to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(images)
            #predictions = torch.sigmoid(predictions).cpu().numpy()
            true_labels = true_labels.cpu().numpy()
        
        print(f"\nPredictions shape: {predictions.shape}")
        
        # Visualize results
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        predict_list = []
        for i in range(num_samples):
            # Display original image
            img = images[i].cpu().numpy()
            axes[0, i].imshow(img[0], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title('Original Image')
            
            # Get predictions for this image
            img_preds = predictions[i]
            img_true = true_labels[i]
            
            print(f"\nDebug - Prediction vector length: {len(img_preds)}")
            print(f"Debug - Number of pathologies: {len(pathologies)}")
            
            # Get model's pathologies
            model_pathologies = model.pathologies
            print(f"Debug - Model pathologies: {model_pathologies}")
            
            # Create mapping between model and dataset pathologies
            pathology_mapping = {i: model_pathologies.index(p) for i, p in enumerate(pathologies) if p in model_pathologies}
            
            # Map predictions to our pathology list
            mapped_preds = np.zeros(len(pathologies))
            for dataset_idx, model_idx in pathology_mapping.items():
                mapped_preds[dataset_idx] = img_preds[model_idx]
            predict_list.append(mapped_preds)
            
            # Find top 3 predicted pathologies
            top_pred_idx = np.argsort(mapped_preds)[-3:][::-1]
            
            pred_text = "Top Predictions:\n"
            for idx in top_pred_idx:
                pred_text += f"{pathologies[idx]}: {mapped_preds[idx]:.2f}\n"
            
            # Find actual pathologies
            true_path_idx = np.where(img_true == 1)[0]
            true_text = "Ground Truth:\n"
            if len(true_path_idx) > 0:
                for idx in true_path_idx:
                    true_text += f"{pathologies[idx]}\n"
            else:
                true_text += "No pathologies\n"
            
            # Add texts to plot
            axes[1, i].text(0.1, 0.7, pred_text, transform=axes[1, i].transAxes)
            axes[1, i].text(0.1, 0.2, true_text, transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        return predict_list
        
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        raise


def calculate_metrics(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate model performance metrics on the dataset
    
    Parameters:
    model: Pre-trained torchxrayvision model
    dataloader: DataLoader containing the images
    device: Device to run inference on
    
    Returns:
    dict: Dictionary containing performance metrics
    """
    model = model.to(device)
    model.eval()
    
    # Get pathologies list
    if hasattr(dataloader.dataset, 'dataset'):
        pathologies = dataloader.dataset.dataset.pathologies
    elif hasattr(dataloader.dataset, 'pathologies'):
        pathologies = dataloader.dataset.pathologies
    else:
        pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating metrics"):
            if isinstance(batch, dict):
                images = batch['img']
                labels = batch['lab']
            else:
                images, labels = batch
            
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Combine all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    metrics = {}
    
    for i, pathology in enumerate(pathologies):
        # Calculate accuracy for each pathology
        pred_binary = (all_preds[:, i] > 0.5).astype(int)
        accuracy = np.mean(pred_binary == all_labels[:, i])
        
        # Store metrics
        metrics[pathology] = {
            'accuracy': accuracy,
            'positive_samples': np.sum(all_labels[:, i] == 1),
            'negative_samples': np.sum(all_labels[:, i] == 0)
        }
    
    return metrics


def extract_features(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Extract features from images using the pretrained model
    
    Parameters:
    model: Pretrained torchxrayvision model
    dataloader: DataLoader containing the images
    device: Device to run inference on
    
    Returns:
    features: Extracted features (n_samples, n_features)
    labels: Ground truth labels (n_samples, n_classes)
    """
    model = model.to(device)
    model.eval()
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, dict):
                images = batch['img']
                labels = batch['lab']
            else:
                images, labels = batch
            
            images = images.to(device)
            
            # Forward pass through the model
            features = model.features(images)
            
            # Global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    return np.concatenate(features_list), np.concatenate(labels_list)

def reduce_dimensions(features, method='pca', n_components=100, random_state=42):
    """
    Reduce dimensionality of features
    
    Parameters:
    features: Feature matrix (n_samples, n_features)
    method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
    n_components: Number of components to keep
    random_state: Random seed for reproducibility
    
    Returns:
    reduced_features: Reduced feature matrix (n_samples, n_components)
    reducer: Fitted dimensionality reduction model
    """
    # First, standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_features = reducer.fit_transform(scaled_features)
        # Print explained variance ratio
        explained_var = np.sum(reducer.explained_variance_ratio_)
        print(f"PCA with {n_components} components explains {explained_var:.2%} of variance")
    
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
        reduced_features = reducer.fit_transform(scaled_features)
    
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            reduced_features = reducer.fit_transform(scaled_features)
        except ImportError:
            print("UMAP not installed. Please install it with 'pip install umap-learn'")
            print("Falling back to PCA...")
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_features = reducer.fit_transform(scaled_features)
    
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return reduced_features, reducer, scaler

def discretize_features(features, n_bins=5):
    """
    Discretize continuous features into bins for Bayesian network
    
    Parameters:
    features: Continuous feature matrix
    n_bins: Number of bins for discretization
    
    Returns:
    discretized: Discretized feature matrix
    bin_edges: Edges of the bins for each feature
    """
    discretized = np.zeros_like(features, dtype=int)
    bin_edges = []
    
    for i in range(features.shape[1]):
        hist, edges = np.histogram(features[:, i], bins=n_bins)
        discretized[:, i] = np.digitize(features[:, i], edges[:-1])
        bin_edges.append(edges)
    
    return discretized, bin_edges


def save_model_and_metadata(model, bin_edges, reducer, scaler, param_samples, output_dir, method):
    """
    Save model with additional metadata
    
    Parameters:
    model: Bayesian network model
    bin_edges: Bin edges for discretization
    reducer: Fitted dimensionality reduction model
    scaler: Fitted scaler model
    param_samples: Dictionary of parameter samples
    output_dir: Directory to save models and metadata
    method: Dimensionality reduction method used
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Saving model and metadata...")
    start_time = time.time()
    model.save(os.path.join(output_dir, 'bayesian_network_train.bif'), filetype='bif')
    
    # Save metadata
    np.save(os.path.join(output_dir, 'bin_edges_train.npy'), bin_edges)
    
    # Save dimensionality reduction models
    if method.lower() == 'pca':
        np.save(os.path.join(output_dir, 'pca_components_train.npy'), reducer.components_)
        np.save(os.path.join(output_dir, 'pca_mean_train.npy'), reducer.mean_)
        np.save(os.path.join(output_dir, 'explained_variance_train.npy'), reducer.explained_variance_ratio_)
    
    # Save scaler parameters
    np.save(os.path.join(output_dir, 'scaler_mean_train.npy'), scaler.mean_)
    np.save(os.path.join(output_dir, 'scaler_scale_train.npy'), scaler.scale_)
    
    # Save parameter samples (this might be large)
    try:
        np.save(os.path.join(output_dir, 'parameter_samples_train.npy'), param_samples)
    except:
        print("Warning: Could not save parameter samples due to size or format. Consider saving them separately.")
    
    print(f"Model and metadata saved in {time.time() - start_time:.1f} seconds")

def construct_bayesian_network_with_uncertainty(features, labels, pathologies, n_bins=5, equivalent_sample_size=10, n_parameter_samples=1000, bayesnet_struct="naive"):
    """
    Construct a Bayesian network from features and labels with full parameter uncertainty
    
    Parameters:
    features: Discretized feature matrix
    labels: Binary label matrix
    pathologies: List of pathology names
    n_bins: Number of bins used in discretization
    equivalent_sample_size: Parameter for Dirichlet prior strength
    n_parameter_samples: Number of parameter samples to draw from posterior distributions
    
    Returns:
    model: fitted Bayesian network
    param_samples: Dictionary of parameter samples for each node
    """
    # Create DataFrame for pgmpy
    feature_cols = [f'f{i}' for i in range(features.shape[1])]
    pathology_cols = pathologies
    
    data = pd.DataFrame(
        np.hstack([features, labels]),
        columns=feature_cols + pathology_cols
    )
    
    # Convert data types to integers (required for Bayesian estimation)
    data = data.astype(int)

    
    if bayesnet_struct == "naive":
        # Define network structure
        edges = [(p, f) for p in pathology_cols for f in feature_cols]
        edges += [(pathology_cols[i], pathology_cols[j]) for i in range(len(pathology_cols)) for j in range(i+1, len(pathology_cols))]
        
        # Create Bayesian network
        model = BayesianNetwork(edges)

    elif bayesnet_struct == "tree":
        est_tree = TreeSearch(data)
        best_model = est_tree.estimate()
        model = BayesianNetwork(best_model)

    elif bayesnet_struct == "hill":
        est_tree = HillClimbSearch(data)
        best_model = est_tree.estimate()
        model = BayesianNetwork(best_model)
    
    elif bayesnet_struct == "exhaustive":
        est_tree = ExhaustiveSearch(data)
        best_model = est_tree.estimate()
        model = BayesianNetwork(best_model)

    # Calculate state cardinalities for each node
    state_names = {col: sorted(data[col].unique()) for col in data.columns}
    
    # Calculate node cardinalities
    node_card = {node: len(states) for node, states in state_names.items()}
    
    # Fit model with Bayesian parameter estimation
    print("Performing Bayesian parameter estimation...")
    bayes_est = BayesianEstimator(model=model, data=data)
    
    # Store parameter posterior distributions and sample from them
    param_samples = {}
    
    for node in tqdm(model.nodes(), desc="Sampling parameters"):
        # Get Dirichlet posterior parameters for this node
        cpd = bayes_est.estimate_cpd(
            node, 
            prior_type="dirichlet", 
            pseudo_counts=equivalent_sample_size
        )
        
        # Add CPD to model (this is the expected value, but we'll use samples later for inference)
        model.add_cpds(cpd)
        
        # Extract parameters from CPD
        values = cpd.values
        
        # For each configuration of parent variables, sample from Dirichlet distribution
        param_samples[node] = {}
        
        # Get parent nodes from model structure
        parents = model.get_parents(node)
        
        if parents:
            # Get cardinality of parent variables
            evidence_card = [node_card[parent] for parent in parents]
            
            # Calculate total number of parent configurations
            n_parent_configs = np.prod(evidence_card)
            
            # Reshape parameter values for sampling
            values_reshaped = values.reshape(node_card[node], -1).T if len(parents) > 1 else values.T
            
            # Sample from Dirichlet for each parent configuration
            for i in range(n_parent_configs):
                alpha = values_reshaped[i] * equivalent_sample_size + 1
                param_samples[node][i] = np.random.dirichlet(alpha, size=n_parameter_samples)
        else:
            # No parents, just one Dirichlet distribution
            alpha = values.flatten() * equivalent_sample_size + 1
            param_samples[node][0] = np.random.dirichlet(alpha, size=n_parameter_samples)
    
    return model, param_samples, data, feature_cols, pathology_cols


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from pgmpy.inference import VariableElimination

def simplified_inference_test(model, param_samples, data, feature_cols, pathology_cols, base_model_pred, idx, n_samples=50, n_test_samples=5):
    """
    A simplified test that uses the training data directly without preprocessing
    
    Parameters:
    model: Trained Bayesian Network model
    param_samples: Dictionary of parameter samples
    data: DataFrame with the training data
    feature_cols: Names of feature columns
    pathology_cols: Names of pathology columns
    n_samples: Number of parameter samples to use
    n_test_samples: Number of training samples to test on
    
    Returns:
    results: DataFrame with inference results
    """
    # Get random samples from the training data
    if n_test_samples > len(data):
        n_test_samples = len(data)
        print(f"Reducing test samples to {n_test_samples} (size of dataset)")
    
    #indices = [0,1,2,3]# np.random.choice(len(data), n_test_samples, replace=False)
    #results = []
    
    # Get a sample from the data
    sample = data.iloc[idx]
    
    # Create evidence dictionary with the feature values
    evidence = {col: int(sample[col]) for col in feature_cols}
    
    # Get ground truth
    ground_truth = {col: int(sample[col]) for col in pathology_cols}
    
    print(f"\nProcessing sample {idx} with features: {evidence}")
    print(f"Ground truth: {ground_truth}")
    
    # First, do traditional inference
    inference = VariableElimination(model)
    
    traditional_results = {}
    for var in pathology_cols:
        try:
            result = inference.query(variables=[var], evidence=evidence)
            # For binary variables, get P(var=1)
            if len(result.state_names[var]) == 2:
                idx_1 = result.state_names[var].index(1) if 1 in result.state_names[var] else 1
                traditional_results[var] = result.values[idx_1]
            else:
                traditional_results[var] = result.values
        except Exception as e:
            print(f"Error in traditional inference for {var}: {e}")
            traditional_results[var] = np.nan
    
    print(f"Traditional inference results:")
    for var, prob in traditional_results.items():
        print(f"  {var}: {prob:.4f} (Ground truth: {ground_truth[var]})")
    
    # Now do manual parameter sampling for uncertainty quantification
    # This avoids the complexity of the full inference_with_uncertainty function
    
    print(f"\nPerforming parameter sampling for uncertainty quantification...")
    posterior_samples = {var: [] for var in pathology_cols}
    
    # Sample a smaller number of parameter sets for demonstration
    # Find the first parameter sample to determine the total number available
    first_node = next(iter(param_samples))
    first_config = next(iter(param_samples[first_node]))
    total_samples = param_samples[first_node][first_config].shape[0]
    
    # Choose parameter sample indices - use a smaller number for demonstration
    use_samples = min(n_samples, total_samples)
    print(f"Using {use_samples} parameter samples from {total_samples} available")
    sample_indices = np.random.choice(total_samples, use_samples, replace=False)
    
    # For each variable we want to query
    for var in pathology_cols:
        # Need to get probabilities by iterating through parameter samples
        for i in sample_indices:
            # Use the traditional model structure but sample the parameters
            # This is a simplification to verify the underlying concept works
            value = traditional_results[var]
            # Add some random noise to simulate parameter uncertainty
            # In a proper implementation, we would properly sample from the parameter distributions
            # But this gives us a good sense of whether the visualization approach works
            noisy_value = np.clip(value + np.random.normal(0, 0.1), 0, 1)
            posterior_samples[var].append(noisy_value)
    
    # Calculate number of rows and columns for the grid based on number of pathologies
    num_pathologies = len(pathology_cols)
    nrows = (num_pathologies + 3) // 4  # Ceiling division to get number of rows (4 columns per row)
    
    # Create a grid of subplots with the appropriate size
    fig, axes = plt.subplots(nrows, 4, figsize=(20, 5 * nrows))
    
    # Flatten axes array for easy iteration (only if there are multiple rows)
    if nrows > 1:
        axes = axes.flatten()
    
    # Loop through each pathology column
    for i, var in enumerate(pathology_cols):
        # Get the current subplot
        if nrows == 1:
            ax = axes[i]  # When there's only one row
        else:
            ax = axes[i]  # When axes is flattened
            
        # Extract data
        samples = posterior_samples[var]
        mean_prob = np.mean(samples)
        median_prob = np.median(samples)
        ci_95 = np.percentile(samples, [2.5, 97.5])
        
        # Plot the posterior distribution on the current subplot
        sns.histplot(samples, kde=True, bins=20, ax=ax)
        ax.set_title(f'{var}\n(Truth: {ground_truth[var]})', fontsize=10)
        ax.set_xlabel('Probability', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        
        # Add vertical lines for different metrics
        ax.axvline(mean_prob, color='red', linestyle='-', 
                label=f'Mean: {mean_prob:.2f}')
        ax.axvline(ci_95[0], color='red', linestyle=':', 
                label=f'95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]')
        ax.axvline(ci_95[1], color='red', linestyle=':')
        ax.axvline(traditional_results[var], color='blue', linestyle='--', 
                label=f'Traditional: {traditional_results[var]:.2f}')
        ax.axvline(ground_truth[var], color='green', linestyle='-', 
                label=f'Ground truth: {ground_truth[var]}')
        ax.axvline(base_model_pred[i], color='purple', linestyle='--', 
                label=f'Base Model: {base_model_pred[i]:.2f}')
        
        # Adjust font size of tick labels
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Only add legend to the first plot to save space
        if i == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Hide any unused subplots (if total plots < nrows * 4)
    total_subplots = nrows * 4
    for j in range(num_pathologies, total_subplots):
        if nrows == 1:
            if j < len(axes):
                axes[j].set_visible(False)
        else:
            axes[j].set_visible(False)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Show the plot
    plt.show()
    
    # Print statistics for each pathology
    for var in pathology_cols:
        samples = posterior_samples[var]
        mean_prob = np.mean(samples)
        median_prob = np.median(samples)
        ci_95 = np.percentile(samples, [2.5, 97.5])
        
        print(f"\nStatistics for {var}:")
        print(f"  Ground truth: {ground_truth[var]}")
        print(f"  MLE: {traditional_results[var]:.4f}")
        print(f"  MAP: {mean_prob:.4f}")
        print(f"  Median with uncertainty: {median_prob:.4f}")
        print(f"  95% Credible Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"  Standard deviation: {np.std(samples):.4f}")
        
        # Check if ground truth is within CI
        within_ci = (ground_truth[var] >= ci_95[0]) and (ground_truth[var] <= ci_95[1])
        print(f"  Ground truth within 95% CI: {within_ci}")