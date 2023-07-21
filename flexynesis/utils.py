import pandas as pd
import numpy as np
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import torch
from sklearn.linear_model import LinearRegression
from typing import Dict
import warnings


def plot_dim_reduced(matrix, labels, method='pca', color_type='categorical', scatter_kwargs=None, legend_kwargs=None, figsize=(10, 8)):
    """
    Plots the first two dimensions of the transformed input matrix in a 2D scatter plot,
    with points colored based on the provided labels. The transformation method can be either PCA or UMAP.
    
    This function allows users to control several aspects of the plot such as the figure size, scatter plot properties, and legend properties.

    Args:
        matrix (np.array): Input data matrix (n_samples, n_features).
        labels (list): List of labels (strings or integers).
        method (str): Transformation method ('pca' or 'umap'). Default is 'pca'.
        color_type (str): Type of the color scale ('categorical' or 'numerical'). Default is 'categorical'.
        scatter_kwargs (dict, optional): Additional keyword arguments for plt.scatter. Default is None.
        legend_kwargs (dict, optional): Additional keyword arguments for plt.legend. Default is None.
        figsize (tuple): Size of the figure (width, height). Default is (10, 8).
    """
    
    plt.figure(figsize=figsize)
    
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}
    legend_kwargs = legend_kwargs if legend_kwargs else {}

    # Compute transformation
    if method.lower() == 'pca':
        transformer = PCA(n_components=2)
    elif method.lower() == 'umap':
        transformer = UMAP(n_components=2)
    else:
        raise ValueError("Invalid method. Expected 'pca' or 'umap'")
        
    transformed_matrix = transformer.fit_transform(matrix)

    # Create a pandas DataFrame for easier plotting
    transformed_df = pd.DataFrame(transformed_matrix, columns=[f"{method.upper()}1", f"{method.upper()}2"])

    labels = [-1 if pd.isnull(x) or x in {'nan', 'None'} else x for x in labels]

    # Add the labels to the DataFrame
    transformed_df["Label"] = labels

    if color_type == 'categorical':
        unique_labels = list(set(labels))
        colormap = matplotlib.colormaps["tab20"]

        for i, label in enumerate(unique_labels):
            plt.scatter(
                transformed_df[transformed_df["Label"] == label][f"{method.upper()}1"],
                transformed_df[transformed_df["Label"] == label][f"{method.upper()}2"],
                color=colormap(i),
                label=label,
                **scatter_kwargs
            )

        plt.xlabel(f"{method.upper()} Dimension 1", fontsize=14)
        plt.ylabel(f"{method.upper()} Dimension 2", fontsize=14)
        plt.title(f"{method.upper()} Scatter Plot with Colored Labels", fontsize=18)
        plt.legend(title="Labels", **legend_kwargs)
    elif color_type == 'numerical':
        sc = plt.scatter(transformed_df[f"{method.upper()}1"], transformed_df[f"{method.upper()}2"], 
                         c=labels, **scatter_kwargs)
        plt.colorbar(sc, label='Label')
    plt.show()

def plot_true_vs_predicted(true_values, predicted_values):
    """
    Plots a scatterplot of true vs predicted values, with a regression line and annotated with the Pearson correlation coefficient.

    Args:
        true_values (list or np.array): True values
        predicted_values (list or np.array): Predicted values
    """
    # Calculate correlation coefficient
    corr, _ = pearsonr(true_values, predicted_values)
    corr_text = f"Pearson r: {corr:.2f}"
    
    # Generate scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.5)
    
    # Add regression line
    m, b = np.polyfit(true_values, predicted_values, 1)
    plt.plot(true_values, m*np.array(true_values) + b, color='red')
    
    # Add correlation text
    plt.text(min(true_values), max(predicted_values), corr_text, fontsize=12, ha='left', va='top')
    
    # Add labels and title
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    plt.show()
    
def evaluate_classifier(y_true, y_pred):
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # F1 score (macro)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score (Macro): {f1:.4f}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Full classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred)
    print(report)
    return {"balanced_acc": balanced_acc, "f1_score": f1, "kappa": kappa}

def evaluate_regressor(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "r2": r2, "pearson_corr": pearson_corr[0]}


def remove_batch_effects2(embeddings: pd.DataFrame, 
                         target_variables: Dict[str, torch.Tensor], 
                         batch_variables: Dict[str, torch.Tensor]) -> (pd.DataFrame, Dict[str, LinearRegression]):
    """
    Removes batch effects from embeddings while preserving target variable effects.
    All variables (target and batch) are assumed to be numerical tensors.

    Args:
        embeddings: A DataFrame where each row is a sample and each column is a dimension of the embeddings.
        target_variables: A dictionary where keys are target variable names and values are 1D tensors of target variable values.
        batch_variables: A dictionary where keys are batch variable names and values are 1D tensors of batch variable values.

    Returns:
        A DataFrame with the same structure as embeddings but with batch effects removed.
        A dictionary where keys are names of embedding dimensions and values are linear regression models that transform initial embeddings to corrected embeddings.
    """
    corrected_embeddings = embeddings.copy()
    transformation_models = {}

    for column in embeddings.columns:
        # Identify the indices of samples with no missing labels
        non_na_indices = []
        for var_dict in [target_variables, batch_variables]:
            for var_name, var_values in var_dict.items():
                non_na_indices.append(~np.isnan(var_values.numpy()))
        non_na_indices = np.all(non_na_indices, axis=0)

        # Check if there are enough non-NA samples to fit a linear regression model
        if np.sum(non_na_indices) < 10:
            warnings.warn(f"Skipping batch correction for embedding dimension '{column}' due to insufficient non-NA samples.")
            continue

        # Fit a linear regression model with target variables only on the samples with no missing labels
        target_predictors = np.column_stack([var_values.numpy()[non_na_indices] for var_values in target_variables.values()])
        target_model = LinearRegression()
        target_model.fit(target_predictors, embeddings.loc[non_na_indices, column].values)

        # Get residuals from the target variables model
        target_effects = target_model.predict(target_predictors)
        residuals = embeddings.loc[non_na_indices, column].values - target_effects

        # Fit a new linear regression model with batch variables on residuals
        batch_predictors = np.column_stack([var_values.numpy()[non_na_indices] for var_values in batch_variables.values()])
        batch_model = LinearRegression()
        batch_model.fit(batch_predictors, residuals)

        # Calculate the batch effects and subtract them from the residuals
        batch_effects = batch_model.predict(batch_predictors)
        corrected_residuals = residuals - batch_effects

        # Add back the target effects to get the corrected embeddings
        corrected_embeddings_non_na = target_effects + corrected_residuals

        # Learn a linear transformation from the initial embeddings to the corrected embeddings
        transformation_model = LinearRegression()
        transformation_model.fit(embeddings.loc[non_na_indices, column].values.reshape(-1, 1),
                                 corrected_embeddings_non_na.reshape(-1, 1))

        # Store the transformation model
        transformation_models[column] = transformation_model

        # Apply the transformation to all embeddings
        corrected_embeddings[column] = transformation_model.predict(embeddings[column].values.reshape(-1, 1))

    return corrected_embeddings, transformation_models

def remove_batch_effects(embeddings: pd.DataFrame, 
                         target_variables: Dict[str, torch.Tensor], 
                         batch_variables: Dict[str, torch.Tensor]) -> pd.DataFrame:
    """
    Removes batch effects from embeddings while preserving target variable effects.
    All variables (target and batch) are assumed to be numerical tensors.

    Args:
        embeddings: A DataFrame where each row is a sample and each column is a dimension of the embeddings.
        target_variables: A dictionary where keys are target variable names and values are 1D tensors of target variable values.
        batch_variables: A dictionary where keys are batch variable names and values are 1D tensors of batch variable values.

    Returns:
        A DataFrame with the same structure as embeddings but with batch effects removed.
    """
    corrected_embeddings = embeddings.copy()

    for column in embeddings.columns:
        # Fit a linear regression model with target variables only
        target_predictors = np.column_stack([var_values.numpy() for var_values in target_variables.values()])
        target_model = LinearRegression()
        target_model.fit(target_predictors, embeddings[column].values)

        # Get residuals from the target variables model
        target_effects = target_model.predict(target_predictors)
        residuals = embeddings[column].values - target_effects

        # Fit a new linear regression model with batch variables on residuals
        batch_predictors = np.column_stack([var_values.numpy() for var_values in batch_variables.values()])
        batch_model = LinearRegression()
        batch_model.fit(batch_predictors, residuals)

        # Calculate the batch effects and subtract them from the residuals
        batch_effects = batch_model.predict(batch_predictors)
        corrected_residuals = residuals - batch_effects

        # Add back the target effects to get the corrected embeddings
        corrected_embeddings[column] = target_effects + corrected_residuals

    return corrected_embeddings


def apply_batch_correction(embeddings: pd.DataFrame, transformation_models: Dict[str, LinearRegression]) -> pd.DataFrame:
    """
    Applies learned transformation models to a new set of embeddings to correct batch effects.
    First run remove_batch_effects on training_dataset and then use the output to transform test dataset embeddings

    Args:
        embeddings: A DataFrame where each row is a sample and each column is a dimension of the embeddings.
        transformation_models: A dictionary where keys are names of embedding dimensions and values are linear regression models that transform initial embeddings to corrected embeddings.

    Returns:
        A DataFrame with the same structure as embeddings but with batch effects corrected.
    """
    corrected_embeddings = embeddings.copy()

    for column, model in transformation_models.items():
        # Apply the transformation model to the corresponding embedding dimension
        if column in corrected_embeddings.columns:
            corrected_embeddings[column] = model.predict(embeddings[column].values.reshape(-1, 1))
        else:
            print(f"Warning: transformation model for '{column}' cannot be applied as this column is not in the provided embeddings.")

    return corrected_embeddings


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def remove_batch_associated_variables(data, variable_types, target_dict, batch_dict = None, mi_threshold=0.1):
    """
    Filter the data matrix to keep only the columns that are predictive of the target variables 
    and not predictive of the batch variables.
    
    Args:
        data (pd.DataFrame): The data matrix.
        target_dict (dict): A dictionary of target variables.
        batch_dict (dict): A dictionary of batch variables.
        variable_types (dict): A dictionary of variable types (either "numerical" or "categorical").
        mi_threshold (float, optional): The mutual information threshold for a column to be considered predictive.
                                        Defaults to 0.1.
    
    Returns:
        pd.DataFrame: The filtered data matrix.
    """
    # Convert target and batch tensors to numpy
    target_dict_np = {k: v.numpy() for k, v in target_dict.items()}
    batch_dict_np = {k: v.numpy() for k, v in batch_dict.items()}

    important_features = set()

    # Find important features for target variables
    for var_name, target in target_dict_np.items():
        # Skip if all values are missing
        if np.all(np.isnan(target)):
            continue
            
        # Subset data and target where target is not missing
        not_missing = ~np.isnan(target)
        data_sub = data[not_missing]
        target_sub = target[not_missing]

        if variable_types[var_name] == "categorical":
            clf = RandomForestClassifier()
        else:  # numerical
            clf = RandomForestRegressor()
            
        clf = clf.fit(data_sub, target_sub)
        model = SelectFromModel(clf, prefit=True)
        important_features.update(data.columns[model.get_support()])

    if batch_dict is not None:
        # Compute mutual information for batch variables
        for var_name, batch in batch_dict_np.items():
            # Skip if all values are missing
            if np.all(np.isnan(batch)):
                continue

            # Subset data and batch where batch is not missing
            not_missing = ~np.isnan(batch)
            data_sub = data[not_missing]
            batch_sub = batch[not_missing]

            if variable_types[var_name] == "categorical":
                mi = mutual_info_classif(data_sub, batch_sub)
            else:  # numerical
                mi = mutual_info_regression(data_sub, batch_sub)

            # Remove features with high mutual information with batch variables
            important_features -= set(data.columns[mi > mi_threshold])

    return data[list(important_features)]