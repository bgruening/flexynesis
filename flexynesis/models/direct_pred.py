import torch
from torch import nn
from torch.nn import functional as F
import lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
import os, argparse
from scipy import stats
from functools import reduce

from captum.attr import IntegratedGradients, GradientShap

from ..modules import *

class DirectPred(pl.LightningModule):
    """
    A fully connected network for multi-omics integration with supervisor heads.

    Attributes:
        config (dict): Configuration settings for the model, including learning rates and dimensions.
        dataset: The MultiOmicDataset object containing the data and metadata.
        target_variables (list): A list of target variable names that the model aims to predict.
        batch_variables (list, optional): A list of variables used for batch correction. Defaults to None.
        surv_event_var (str, optional): The name of the survival event variable. Defaults to None.
        surv_time_var (str, optional): The name of the survival time variable. Defaults to None.
        use_loss_weighting (bool, optional): Whether to use loss weighting in the model. Defaults to True.
        device_type (str, optional): Type of device to run the model ('gpu' or 'cpu'). Defaults to None.
    """

    def __init__(self, config, dataset, target_variables, batch_variables = None, 
                 surv_event_var = None, surv_time_var = None, use_loss_weighting = True,
                device_type = None):
        super(DirectPred, self).__init__()
        self.config = config
        self.target_variables = target_variables
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        # both surv event and time variables are assumed to be numerical variables
        # we create only one survival variable for the pair (surv_time_var and surv_event_var)
        if self.surv_event_var is not None and self.surv_time_var is not None:
            self.target_variables = self.target_variables + [self.surv_event_var]
        self.batch_variables = batch_variables
        self.variables = self.target_variables + batch_variables if batch_variables else self.target_variables
        self.feature_importances = {}
        self.use_loss_weighting = use_loss_weighting
        self.device_type = device_type
        
        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for var in self.variables:
                self.log_vars[var] = nn.Parameter(torch.zeros(1))

        self.variable_types = dataset.variable_types
        self.ann = dataset.ann
        self.layers = list(dataset.dat.keys())
        self.input_dims = [len(dataset.features[self.layers[i]]) for i in range(len(self.layers))]

        self.encoders = nn.ModuleList([
            MLP(input_dim=self.input_dims[i],
                # define hidden_dim size relative to the input_dim size
                hidden_dim=int(self.input_dims[i] * self.config['hidden_dim_factor']),
                output_dim=self.config['latent_dim']) for i in range(len(self.layers))])

        if len(self.input_dims) > 1:
            self.fusion_block = nn.Linear(
                in_features=self.config['latent_dim'] * len(self.layers),
                out_features=self.config['latent_dim']
            )
        else:
            self.fusion_block = None
        
        self.MLPs = nn.ModuleDict()  # using ModuleDict to store multiple MLPs
        for var in self.variables:
            if self.variable_types[var] == 'numerical':
                num_class = 1
            else:
                num_class = len(np.unique(self.ann[var]))
            self.MLPs[var] = MLP(input_dim=self.config['latent_dim'],
                                 hidden_dim=self.config['supervisor_hidden_dim'],
                                 output_dim=num_class)

    def forward(self, x_list):
        """
        Forward pass of the DirectPred model.

        Args:
            x_list (list of torch.Tensor): A list of input matrices (omics layers), one for each layer.

        Returns:
            dict: A dictionary where each key-value pair corresponds to the target variable name and its predicted output respectively.
        """
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x))
        embeddings_concat = torch.cat(embeddings_list, dim=1)
        # if multiple embeddings, fuse them 
        embeddings = self.fusion_block(embeddings_concat) if self.fusion_block else embeddings_concat

        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(embeddings)
        return outputs  
    
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the DirectPred model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
    
    def compute_loss(self, var, y, y_hat):
        """
        Computes the loss for a specific variable based on whether the variable is numerical or categorical.
        Handles missing labels by excluding them from the loss calculation.
    
        Args:
            var (str): The name of the variable for which the loss is being calculated.
            y (torch.Tensor): The true labels or values for the variable.
            y_hat (torch.Tensor): The predicted labels or values output by the model.
    
        Returns:
            torch.Tensor: The calculated loss tensor for the variable. If there are no valid labels or values
                          to compute the loss (all are missing), returns a zero loss tensor with gradient enabled.
    
        The method first checks the type of the variable (`var`) from `variable_types`. If the variable is
        numerical, it computes the mean squared error loss. For categorical variables, it calculates the
        cross-entropy loss. The method ensures to ignore any instances where the labels are missing (NaN for
        numerical or -1 for categorical as assumed missing value encoding) when calculating the loss.
        """
        if self.variable_types[var] == 'numerical':
            # Ignore instances with missing labels for numerical variables
            valid_indices = ~torch.isnan(y)
            if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                y_hat = y_hat[valid_indices]
                y = y[valid_indices]
                loss = F.mse_loss(torch.flatten(y_hat), y.float())
            else:
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True) # if no valid labels, set loss to 0
        else:
            # Ignore instances with missing labels for categorical variables
            # Assuming that missing values were encoded as -1
            valid_indices = (y != -1) & (~torch.isnan(y))
            if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                y_hat = y_hat[valid_indices]
                y = y[valid_indices]
                loss = F.cross_entropy(y_hat, y.long())
            else: 
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        return loss
    
    def compute_total_loss(self, losses):
        """
        Computes the total loss from a dictionary of individual losses. This method can compute
        either weighted or unweighted total loss based on the model configuration. If loss weighting
        is enabled and there are multiple loss components, it uses uncertainty-based weighting.
        See Kendall A. et al, https://arxiv.org/abs/1705.07115.
        
        Args:
            losses (dict of torch.Tensor): A dictionary where each key is a variable name and
                                           each value is the loss tensor associated with that variable.
    
        Returns:
            torch.Tensor: The total loss computed across all inputs, either weighted or unweighted.
        
        The method checks if loss weighting is used (`use_loss_weighting`) and if there are multiple
        losses to weight. If so, it computes the weighted sum of losses, where the weight involves
        the exponential of the negative log variance (acting as precision) associated with each loss,
        added to the log variance itself. This approach helps in balancing the contribution of each
        loss component based on its uncertainty. If loss weighting is not used, or there is only one
        loss component, it sums up the losses directly.
        """
        if self.use_loss_weighting and len(losses) > 1:
            # Compute weighted loss for each loss 
            # Weighted loss = precision * loss + log-variance
            total_loss = sum(torch.exp(-self.log_vars[name]) * loss + self.log_vars[name] for name, loss in losses.items())
        else:
            # Compute unweighted total loss
            total_loss = sum(losses.values())
        return total_loss

    def training_step(self, train_batch, batch_idx, log = True):
        """
        Executes one training step using a single batch from the training dataset.

        Args:
            train_batch (tuple): The batch to train on, which includes input data and targets.
            batch_idx (int): Index of the current batch in the sequence.
            log (bool, optional): Whether to log the loss metrics to TensorBoard. Defaults to True.

        Returns:
            torch.Tensor: The total loss computed for the batch.
        """
        
        dat, y_dict, samples = train_batch       
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        outputs = self.forward(x_list)
        losses = {}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var] 
                risk_scores = outputs[var] #output of MLP
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss
            
        total_loss = self.compute_total_loss(losses)
        # add train loss for logging
        losses['train_loss'] = total_loss
        if log:
            self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
    
    def validation_step(self, val_batch, batch_idx, log = True):
        """
        Executes one validation step using a single batch from the validation dataset.

        Args:
            val_batch (tuple): The batch to validate on, which includes input data and targets.
            batch_idx (int): Index of the current batch in the sequence.
            log (bool, optional): Whether to log the loss metrics to TensorBoard. Defaults to True.

        Returns:
            torch.Tensor: The total loss computed for the batch.
        """
        dat, y_dict, samples = val_batch       
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        outputs = self.forward(x_list)
        losses = {}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var] 
                risk_scores = outputs[var] #output of MLP
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss
        total_loss = sum(losses.values())
        losses['val_loss'] = total_loss
        if log:
            self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def predict(self, dataset):
        """
        Evaluate the model on a dataset using batching.

        Args:
            dataset (MultiOmicDataset): dataset containing input matrices for each omics layer.

        Returns:
            dict: Predicted values mapped by target variable names.
        """
        self.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Move the model to the appropriate device

        # Create a DataLoader with a practical batch size
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust the batch size as needed

        predictions = {var: [] for var in self.variables}  # Initialize prediction storage

        # Process each batch
        for batch in dataloader:
            dat, y_dict, samples = batch
            x_list = [dat[x].to(device) for x in dat.keys()]  # Prepare the data batch for processing

            # Perform the forward pass
            outputs = self.forward(x_list)

            # Collect predictions for each variable
            for var in self.variables:
                logits = outputs[var].detach().cpu()  # Raw model outputs (logits)
                
                if dataset.variable_types[var] == 'categorical':
                    probs = torch.softmax(logits, dim=1).numpy() # class probabilities between 0 and 1
                    predictions[var].extend(probs)
                else:
                    predictions[var].extend(logits.numpy()) # return raw output for regression problems
        # Convert lists to arrays 
        predictions = {var: np.array(predictions[var]) for var in predictions}

        return predictions
    
    def transform(self, dataset):
        """
        Transforms the input data into a lower-dimensional representation using trained encoders.

        Args:
            dataset: The dataset containing the input data.

        Returns:
            pd.DataFrame: DataFrame containing the transformed data.
        """
        self.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Move the model to the appropriate device

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust the batch size as needed

        embeddings_list = []  # Initialize a list to collect all batch embeddings
        sample_names = []  # List to collect sample names

        # Process each batch
        for batch in dataloader:
            dat, _, samples = batch
            batch_embeddings = []
            # Process each input matrix with its corresponding Encoder
            for i, x in enumerate(dat.values()):
                x = x.to(device)  # Move data to GPU
                encoded_x = self.encoders[i](x)  # Transform data using the corresponding encoder
                batch_embeddings.append(encoded_x)
            
            # Concatenate all embeddings from the current batch
            embeddings_batch_concat = torch.cat(batch_embeddings, dim=1)
            # if multiple embeddings, fuse them 
            embeddings_batch = self.fusion_block(embeddings_batch_concat) if self.fusion_block else embeddings_batch_concat

            embeddings_list.append(embeddings_batch.detach().cpu())  # Move tensor back to CPU and detach
            sample_names.extend(samples)  # Collect sample names

        # Concatenate all batch embeddings into one tensor
        embeddings_concat = torch.cat(embeddings_list, dim=0)

        # Converting tensor to numpy array and then to DataFrame
        embeddings_df = pd.DataFrame(embeddings_concat.numpy(), 
                                     index=sample_names,  # Set DataFrame index to sample names
                                     columns=[f"E{dim}" for dim in range(embeddings_concat.shape[1])])
        return embeddings_df
        
    # Adaptor forward function for captum integrated gradients or gradient shap 
    def forward_target(self, *args):
        input_data = list(args[:-2])  # one or more tensors (one per omics layer)
        target_var = args[-2]  # target variable of interest
        steps = args[-1]  # number of steps/samples for IntegratedGradients().attribute or GradientShap.attribute 
        outputs_list = []
        for i in range(steps):
            # get list of tensors for each step into a list of tensors
            x_step = [input_data[j][i] for j in range(len(input_data))]
            out = self.forward(x_step)
            outputs_list.append(out[target_var])
        return torch.cat(outputs_list, dim = 0)

    def compute_feature_importance(self, dataset, target_var, method="IntegratedGradients", steps_or_samples=5, batch_size=64):
        """
        Computes the feature importance for each variable in the dataset using either Integrated Gradients or Gradient SHAP.

        Args:
            dataset: The dataset object containing the features and data.
            target_var (str): The target variable for which feature importance is calculated.
            method (str, optional): The attribution method to use ("IntegratedGradients" or "GradientShap").
                                    Defaults to "IntegratedGradients".
            steps_or_samples (int, optional): Number of steps for Integrated Gradients or samples for Gradient SHAP.
                                              Defaults to 5.
            batch_size (int, optional): The size of the batch to process the dataset. Defaults to 64.

        Returns:
            pd.DataFrame: A DataFrame containing feature importances across different variables and data modalities.
        """
        device = torch.device("cuda" if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.to(device)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Choose the attribution method dynamically
        if method == "IntegratedGradients":
            explainer = IntegratedGradients(self.forward_target)
        elif method == "GradientShap":
            explainer = GradientShap(self.forward_target)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'IntegratedGradients' or 'GradientShap'.")

        # Handle target class (numerical vs categorical)
        if dataset.variable_types[target_var] == 'numerical':
            num_class = 1
        else:
            num_class = len(np.unique([y[target_var] for _, y, _ in dataset]))

        aggregated_attributions = [[] for _ in range(num_class)]
        for batch in dataloader:
            dat, _, _ = batch
            x_list = [dat[x].to(device) for x in dat.keys()]
            input_data = tuple([data.unsqueeze(0).requires_grad_() for data in x_list])
            
            if method == 'IntegratedGradients':
                baseline = tuple(torch.zeros_like(x) for x in input_data)
            elif method == 'GradientShap': # provide multiple baselines for Gr.Shap
                baseline = tuple(
                    torch.cat([torch.zeros_like(x) for _ in range(steps_or_samples)], dim=0)
                    for x in input_data
                )
            if num_class == 1:
                # returns a tuple of tensors (one per data modality)
                if method == 'IntegratedGradients':
                    attributions = explainer.attribute(input_data, baseline, 
                                                 additional_forward_args=(target_var, steps_or_samples), 
                                                 n_steps=steps_or_samples)
                elif method == 'GradientShap':
                    attributions = explainer.attribute(input_data, baseline, 
                                                 additional_forward_args=(target_var, steps_or_samples), 
                                                 n_samples=steps_or_samples)
                aggregated_attributions[0].append(attributions)
            else:
                for target_class in range(num_class):
                    # returns a tuple of tensors (one per data modality)
                    if method == 'IntegratedGradients':
                        attributions = explainer.attribute(input_data, baseline, 
                                                           additional_forward_args=(target_var, steps_or_samples), 
                                                           target=target_class,
                                                           n_steps=steps_or_samples)
                    elif method == 'GradientShap':
                        attributions = explainer.attribute(input_data, baseline, 
                                                           additional_forward_args=(target_var, steps_or_samples), 
                                                           target=target_class,
                                                           n_samples=steps_or_samples)
                    aggregated_attributions[target_class].append(attributions)
        # Post-process attributions
        layers = list(dataset.dat.keys())
        num_layers = len(layers)
        processed_attributions = []
        for class_idx in range(len(aggregated_attributions)):
            class_attr = aggregated_attributions[class_idx]
            layer_attributions = []
            for layer_idx in range(num_layers):
                layer_tensors = [batch_attr[layer_idx] for batch_attr in class_attr]
                attr_concat = torch.cat(layer_tensors, dim=1)
                layer_attributions.append(attr_concat)
            processed_attributions.append(layer_attributions)

        abs_attr = [[torch.abs(a).cpu() for a in attr_class] for attr_class in processed_attributions]
        imp = [[a.mean(dim=1) for a in attr_class] for attr_class in abs_attr]
        self.to('cpu')

        # Combine results into a DataFrame
        df_list = []
        for i in range(num_class):
            for j in range(len(layers)):
                features = dataset.features[layers[j]]
                importances = imp[i][j][0].detach().numpy()
                target_class_label = dataset.label_mappings[target_var].get(i) if target_var in dataset.label_mappings else ''
                df_list.append(pd.DataFrame({'target_variable': target_var, 
                                             'target_class': i, 
                                             'target_class_label': target_class_label,
                                             'layer': layers[j], 
                                             'name': features, 
                                             'importance': importances}))    
        df_imp = pd.concat(df_list, ignore_index=True)
        self.feature_importances[target_var] = df_imp

