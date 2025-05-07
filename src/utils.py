import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

def eval(y_true, y_pred, Plot=False):
    """
    Evaluate a binary classification model using several standard metrics.

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted binary labels.
        Plot (bool): If True, displays the ROC curve (default is False).
        
    Returns:
        float: Accuracy score.
        float: Precision score.
        float: Recall score.
        float: F1 score.
        float: ROC AUC score.
        float: PR AUC score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc_ = auc(fpr, tpr)

    if Plot:
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_))
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Ligne de référence
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return accuracy, precision, recall, f1, roc_auc, pr_auc

def status_peptides(correlation, threshold, scaling_factor=0.7, dynamic=False):
    """
    Determine the status of each peptide based on its correlation with others.

    A peptide is marked as 'bon' (good) if it has at least one correlation greater than or equal to the threshold.
    Otherwise, it is marked as 'mauvais' (bad). Optionally, the threshold can be scaled dynamically based on
    the maximum value in the correlation matrix.

    Parameters:
        correlation (pd.DataFrame): Correlation matrix (e.g., Spearman) between peptides.
        threshold (float): Static threshold to determine if correlation is good.
        scaling_factor (float): Scaling factor applied to the max correlation value if dynamic=True.
        dynamic (bool): Whether to compute the threshold dynamically.

    Returns:
        pd.DataFrame: A DataFrame with two columns: "Peptide" and "Status" ('bon' or 'mauvais').
        bool: True if all peptides are to be removed (none pass the threshold), False otherwise.
    """
    peptides_status = {}

    # Replace diagonal values with 0 to avoid self-correlation
    tmp = np.array(correlation)
    np.fill_diagonal(tmp, 0)
    max_threshold = tmp.max()

    # Use dynamic threshold if enabled
    if dynamic:
        threshold = scaling_factor * max_threshold

    remove_count = 0

    # Determine status for each peptide
    for peptide in correlation.columns:
        corr = correlation[peptide].drop(peptide)  # Drop self-correlation
        good_count = (corr >= threshold).sum()

        if good_count == 0:
            remove_count += 1

        peptides_status[peptide] = "bon" if good_count >= 1 else "mauvais"

    # If no peptides pass the threshold, set flag to remove all
    REMOVE_ALL = remove_count == correlation.shape[0]

    df_status = pd.DataFrame(peptides_status.items(), columns=["Peptide", "Status"])

    return df_status, REMOVE_ALL


def compute_correlation(data, threshold, protein_name="ProteinName",
                        total_area_name="TotalArea_light(raw_data)", peptide_name="PrecursorIonNamelight",
                        dynamic=False):
    """
    Compute the Spearman correlation matrix for peptides of a given protein, and evaluate their status.

    Parameters:
        data (pd.DataFrame): Subset of the full dataset containing only one protein.
        threshold (float): Correlation threshold to classify peptide status.
        protein_name (str): Column name identifying proteins.
        total_area_name (str): Column name for intensity values (e.g., TotalArea).
        peptide_name (str): Column name identifying peptides.
        dynamic (bool): Whether to apply a dynamic threshold.

    Returns:
        pd.DataFrame: A DataFrame with "ProteinName", "Peptide", and "Status" columns
                      (empty if all peptides are below the threshold).
    """
    list_of_arrays = []

    # Group intensity values per peptide
    for pept_name in data[peptide_name].unique():
        total_area_light = data[data[peptide_name] == pept_name][total_area_name]
        list_of_arrays.append(total_area_light)

    peptide_names = data[peptide_name].unique()

    # Build DataFrame with peptides as columns and replicate intensities as rows
    df = pd.DataFrame()
    for i, array in enumerate(list_of_arrays):
        tmp_array = np.array(array)
        df[f"{peptide_names[i]}"] = tmp_array

    # Compute correlation matrix
    correlation = df.corr(method='spearman')

    # Get protein name for identification
    df_prot = data[protein_name].iloc[0]

    # Get peptide status based on correlation
    df_status, REMOVE_ALL = status_peptides(correlation, threshold, dynamic=dynamic)

    # Return empty DataFrame if no peptides pass the threshold
    if REMOVE_ALL:
        return pd.DataFrame()

    return pd.concat([pd.Series([df_prot] * len(df_status), name="ProteinName"), df_status], axis=1)


def compute_status(data, threshold,
                   protein_name="ProteinName", total_area_name="TotalArea_light(raw_data)",
                   peptide_name="PrecursorIonNamelight", dynamic=False):
    """
    Compute the status ('bon' or 'mauvais') for each peptide across all proteins in the dataset.

    Parameters:
        data (pd.DataFrame): The full dataset containing multiple proteins and peptides.
        threshold (float): Correlation threshold for determining peptide status.
        protein_name (str): Column name identifying proteins.
        total_area_name (str): Column name for intensity values (e.g., TotalArea).
        peptide_name (str): Column name identifying peptides.
        dynamic (bool): Whether to compute a dynamic threshold for each protein.

    Returns:
        pd.DataFrame: Combined status for all peptides across all proteins with columns:
                      "ProteinName", "Peptide", "Status".
    """
    # Copy and clean peptide names (remove prefixes and brackets)
    copy_data = data.copy()
    copy_data[peptide_name] = copy_data[peptide_name].str.replace(r'^pep_|\[.*?\]', '', regex=True)

    # Get all unique protein names
    prot_names = copy_data[protein_name].unique()

    final_status = pd.DataFrame()

    # Compute status for each protein individually
    for prot_name in prot_names:
        tmp_data = copy_data[copy_data[protein_name] == prot_name]
        status = compute_correlation(tmp_data, threshold=threshold,
                                     protein_name=protein_name,
                                     total_area_name=total_area_name,
                                     peptide_name=peptide_name,
                                     dynamic=dynamic)
        final_status = pd.concat([final_status, status], ignore_index=True)

    return final_status

def balance_classes_with_oversampling(df, input_column='sequence', target_column='quantotypic',
                                      loss_type='hard', threshold=0.5):
    """
    Balance class distribution in a DataFrame using RandomOverSampler.

    For 'hard' loss types, the target labels are already binary or discrete.
    For 'soft' loss types, where the target is continuous (e.g., predicted probabilities), 
    a temporary binary column is created based on a threshold to perform oversampling.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        input_column (str): Name of the column containing input sequences.
        target_column (str): Name of the column containing target values.
        loss_type (str): Either 'hard' or 'soft', depending on whether the loss function 
                         uses discrete or continuous targets.
        threshold (float): Threshold used to binarize the target in the 'soft' case.

    Returns:
        pd.DataFrame: A new DataFrame with balanced classes using oversampling.
    """
    ros = RandomOverSampler(random_state=42)

    if loss_type == 'soft':
        # Create a temporary binary column based on the threshold
        df_temp = df.copy()
        df_temp['target_binary'] = (df_temp[target_column] > threshold).astype(int)

        # Apply oversampling based on the temporary binary column
        balanced_df, _ = ros.fit_resample(df_temp, df_temp['target_binary'])

        # Remove the temporary column before returning
        balanced_df = balanced_df.drop(columns=['target_binary'])
        return balanced_df
    else:
        # 'Hard' case: targets are already binary
        X = df[[input_column]].values  # Must be 2D for RandomOverSampler
        y = df[target_column].values
        X_resampled, y_resampled = ros.fit_resample(X, y)
        balanced_df = pd.DataFrame({
            input_column: X_resampled.flatten(),
            target_column: y_resampled
        })
        return balanced_df

def write_into_json(dict, filename):
    """
    Write a dictionary to a JSON file.

    Parameters:
        dict (dict): The dictionary to write into the file.
        filename (str): The path to the JSON file to create.
        
    Returns:
        None
    """
    import json
    with open(filename, 'w') as f:
        json.dump(dict, f)