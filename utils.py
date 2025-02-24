import os
from random import sample
from turtle import st
import numpy as np
import torch
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import pickle


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime

cuda = True if torch.cuda.is_available() else False  # use GPU if available


def cal_sample_weight(
    labels: np.ndarray, num_class: int, use_sample_weight: bool = True
):
    """Calculate sample weights for each sample in the dataset.

    Parameters
    ----------
    labels : numpy array
        Labels of the dataset.
    num_class : int
        Number of classes.
    use_sample_weight : bool, optional
        Whether to use sample weights. The default is True.

    Returns
    -------
    sample_weight : numpy array
        Sample weights for each sample in the dataset.
    """
    if not use_sample_weight:  # if not use sample weights, return uniform weights
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    """Convert a tensor of labels to one-hot representation.

    Parameters
    ----------
    y : torch tensor
        Labels of the dataset.
    num_dim : int
        Number of classes.

    Returns
    -------
    y_onehot : torch tensor
        One-hot representation of the labels.
    """
    y_onehot = torch.zeros(y.shape[0], num_dim)  # initialize one-hot tensor
    y_onehot.scatter_(1, y.view(-1, 1), 1)  # fill in the one-hot tensor

    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Calculate cosine distance between two tensors.

    Parameters
    ----------
    x1 : torch tensor
        First tensor.
    x2 : torch tensor, optional
        Second tensor. The default is None.
    eps : float, optional
        Epsilon is the minimum value to avoid division by zero. The default is 1e-8.

    Returns
    -------
    torch tensor
        Cosine distance between two tensors.
    """
    x2 = x1 if x2 is None else x2  # if x2 is not provided, use x1 as x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)  # calculate the norm of x1
    w2 = (
        w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    )  # if x2 is not provided, use x1 as x2
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    """Convert a tensor to sparse tensor.

    Parameters
    ----------
    x : torch tensor
        Input tensor.

    Returns
    -------
    sparse_tensortype
        Sparse tensor.
    """
    x_typename = torch.typename(x).split(".")[-1]  # get the type of the input tensor
    sparse_tensortype = getattr(torch.sparse, x_typename)  # get the sparse tensor type
    indices = torch.nonzero(x)  # get the indices of non-zero elements
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(
            *x.shape
        )  # return a sparse tensor with all elements zeros
    indices = indices.t()  # transpose the indices
    values = x[
        tuple(indices[i] for i in range(indices.shape[0]))
    ]  # get the values of non-zero elements
    return sparse_tensortype(indices, values, x.size())  # return a sparse tensor


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    """Calculate the parameter for the adjacency matrix.

    Parameters
    ----------
    edge_per_node : int
        Number of edges per node.
    data : numpy array
        Data matrix.
    metric : str, optional
        Metric to calculate the distance between two data points. The default is "cosine".

    Returns
    -------
    float
        Parameter for the adjacency matrix.
    """
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(
        dist.reshape(
            -1,
        )
    ).values[edge_per_node * data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())
    # return np.asscalar(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    """Generate a graph from a pairwise distance matrix.

    Parameters
    ----------
    dist : torch tensor
        Pairwise distance matrix.
    parameter : float
        Parameter for the adjacency matrix.
    self_dist : bool, optional
        Whether the input is a self-distance matrix. The default is True.

    Returns
    -------
    g : torch tensor
        Adjacency matrix.
    """
    if self_dist:
        assert (
            dist.shape[0] == dist.shape[1]
        ), "Input is not pairwise dist matrix"  # check if the input is a self-distance matrix
    g = (dist <= parameter).float()  # generate the adjacency matrix
    if self_dist:  # if the input is a self-distance matrix
        diag_idx = np.diag_indices(g.shape[0])  # get the indices of diagonal elements
        g[diag_idx[0], diag_idx[1]] = 0  # set the diagonal elements to zero

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    """Generate an adjacency matrix from a data matrix.

    Parameters
    ----------
    data : torch tensor
        Data matrix.
    parameter : float
        Parameter for the adjacency matrix.
    metric : str, optional
        Metric to calculate the distance between two data points. The default is "cosine".

    Returns
    -------
    adj : torch tensor
        Adjacency matrix.
    """
    assert (
        metric == "cosine"
    ), "Only cosine distance implemented"  # check if the metric is cosine distance
    dist = cosine_distance_torch(data, data)  # calculate the pairwise distance matrix
    g = graph_from_dist_tensor(
        dist, parameter, self_dist=True
    )  # generate the adjacency matrix
    if metric == "cosine":  # if the metric is cosine distance
        adj = 1 - dist  # calculate the adjacency matrix
    else:
        raise NotImplementedError  # raise error if the metric is not cosine distance
    adj = adj * g  # set the non-adjacent elements to zero
    adj_T = adj.transpose(
        0, 1
    )  # transpose the adjacency matrix to get the symmetric adjacency matrix
    I = torch.eye(adj.shape[0])  # generate an identity matrix
    if cuda:  # if cuda is available
        I = I.cuda()  # move the identity matrix to cuda
    adj = (
        adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    )  # set the non-adjacent elements to zero
    adj = F.normalize(adj + I, p=1)  # normalize the adjacency matrix
    adj = to_sparse(adj)  # convert the adjacency matrix to sparse matrix

    return adj


def gen_test_adj_mat_tensor(
    data: torch.tensor, trte_idx: dict, parameter: float, metric: str = "cosine"
) -> torch.tensor:
    """Generate an adjacency matrix from a data matrix.

    Parameters
    ----------
    data : torch tensor
        Data matrix.
    trte_idx : dict
        Dictionary of training and test indices.
    parameter : float
        Parameter for the adjacency matrix.
    metric : str, optional
        Metric to calculate the distance between two data points. The default is "cosine".

    Returns
    -------
    adj : torch tensor
        Adjacency matrix.
    """
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr  # retain selected edges

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def save_model_dict(folder: str, model_dict: dict) -> dict:
    """Save the model dictionary.

    Parameters
    ----------
    folder : str
        Path to the folder to save the model dictionary.
    model_dict : dict
        Model dictionary.

    Returns
    -------
    None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(
            model_dict[module].state_dict(), os.path.join(folder, module + ".pth")
        )
    return


def load_model_dict(folder: str, model_dict: dict) -> dict:
    """Load the model dictionary.

    Parameters
    ----------
    folder : str
        Path to the folder containing the model dictionary.
    model_dict : dict
        Model dictionary.

    Returns
    -------
    model_dict : dict
        Model dictionary.
    """
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module + ".pth")):
            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(
                torch.load(
                    os.path.join(folder, module + ".pth")
                    # map_location="cuda:{:}".format(torch.cuda.current_device()),
                )
            )
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


# data preprocessing

# include the function input type


def import_process_datafile(
    raw_data_file_path: str, columns_to_drop: list, index_column: str
) -> pd.DataFrame:
    """Import and process the data file

    Parameters
    ----------
    raw_data_file_path : str
        Path to the raw data file.
    columns_to_drop : list
        List of columns to drop.
    index_column : str
        Column to set as index.

    Returns
    -------
    df_T : pandas dataframe
    """
    df = pd.read_csv(raw_data_file_path, sep=",")
    df = df.drop(columns=columns_to_drop)  # drop gene_num column
    df.set_index(index_column, inplace=False)  # set gene_name as index
    df_T = df.T
    df_T.columns = df_T.iloc[0:1].values[0]  # set the first row as column names
    df_T = df_T.iloc[1:, :]  # drop the first row

    return df_T


def get_label_dict(
    labels_path: str, col_list: list = ["sample_name", "disease"]
) -> dict:
    """Get the label dictionary from the labels file.

    Parameters
    ----------
    labels_path : str
        Path to the labels file.
    col_list : list, optional
        List of columns to use as key and value. The default is ["sample_name", "disease"].

    Returns
    -------
    label_dict : dict
        Dictionary of labels.
    """
    df_labels = pd.read_csv(labels_path)
    df_sample_names_labels = df_labels[col_list]
    df_sample_names_labels.set_index(col_list[0], inplace=True)
    label_dict = df_sample_names_labels[col_list[1]].to_dict()

    return label_dict


def create_dict_from_col(df: pd.DataFrame, key_col: str, value_col: str) -> dict:
    """Create a dictionary of labels from a dataframe.

    Parameters
    ----------
    df_path : str
        Path to the dataframe.
    key_col : str
        Column to use as key.
    value_col : str
        Column to use as value.

    Returns
    -------
    dict
        Dictionary of labels.
    """
    # df = pd.read_csv(df_path)
    df["fetal_near_miss"] = df["fetal_near_miss"].apply(
        lambda x: str(int(x)) if not pd.isnull(x) else x
    )  # convert to string
    return {key: value for key, value in zip(df[key_col], df[value_col])}


def labels_to_startify_data(
    stratify: bool,
    labels_df: pd.DataFrame,
    strat_col_list: list,
    y: pd.DataFrame,
    labels_dict: dict,
):
    """Create labels from the list of columns to stratify the data

    Args:
        stratify (bool): Whether to stratify the data
        labels_df (pd.DataFrame): DataFrame containing the labels
        strat_col_list (list): List of columns to use for stratification
        y (pd.DataFrame): DataFrame containing the target variable
        labels_dict (dict): Dictionary to store the labels

    Returns:
        tuple: y_strat and strat_dict if stratify is True, else None
    """
    if stratify:
        labels_df["strat_col"] = labels_df[strat_col_list].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )
        y_strat = labels_df["strat_col"]
        strat_dict = create_dict_from_col(labels_df, "Maternal_woman_id", "strat_col")
        return y_strat, strat_dict
    else:
        return y, labels_dict


def save_feat_name(
    j: int,
    df: pd.DataFrame,
    data_folder_path: str,
    i: int = None,
    CV: bool = False,
    stratify: bool = True,
) -> pd.DataFrame:
    """
    Save the feature names as csv files

    Parameters
    ----------
    j : int
        Omic number.
    df : pandas dataframe
        Dataframe containing the data.
    data_folder_path : str
        Path to the data folder.
    i : int
        In case oc CV, split number.
    CV : bool, optional
        Whether to use cross validation. The default is False.
    stratify : bool, optional
        Whether the stratified sampling has been used or not. The default is True.

    Returns
    -------
    df_feat : pandas dataframe
        Dataframe containing the feature names.
    """
    # save the feature names as `j_featname.csv`
    df_feat = pd.DataFrame(
        df.columns.values.tolist(),
        columns=["feature_name"],  # , index=range(len(df.columns)
    )
    if stratify:
        sample_type_path = os.path.join(data_folder_path, "strat")
    else:
        sample_type_path = os.path.join(data_folder_path, "no_strat")
    if CV:
        folder_name = "CV"
        folder_path = os.path.join(sample_type_path, folder_name)
        if not os.path.exists(folder_path):  # if the "CV" folder does not exist
            os.makedirs(folder_path)  # create the folder
        cv_folder = os.path.join(folder_path, "CV_" + str(i))
        if not os.path.exists(cv_folder):  # if the "CV_i" folder does not exist
            os.makedirs(cv_folder)  # create the folder
        file_name = str(j) + "_featname.csv"
        file_path = os.path.join(cv_folder, file_name)
        df_feat.to_csv(file_path, header=None, index=None)

    else:
        folder_name = "NO_CV"
        folder_path = os.path.join(sample_type_path, folder_name)
        if not os.path.exists(folder_path):  # if the "NO_CV" folder does not exist
            os.makedirs(folder_path)  # create the folder

        file_name = str(j) + "_featname.csv"
        file_path = os.path.join(folder_path, file_name)
        df_feat.to_csv(file_path, header=None, index=None)

    return df_feat


# train_test_split function
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    y_strat: np.ndarray,
    test_size: float,
    sample_folder: str,
    labels_folder_path: str,
    n_splits: int,
    CV: bool = False,
    stratify: bool = True,
) -> str:
    """
    Split the data into train and test using the sample ids lists in train, test lists

    Parameters
    ----------
    X : numpy array
        Data matrix.
    y : numpy array
        Labels.
    y_strat : numpy array
        Column wrt which the data is stratified.
    test_size : float
        Test size.
    sample_folder : str
        Path to the sample folder.
    labels_folder_path : str
        Path to the labels folder.
    n_splits : int
        Number of splits.
    CV : bool, optional
        Whether to use cross validation. The default is False.
    stratify : bool, optional
        Whether to use stratified sampling. The default is True.

    Returns
    -------
    train_test_folder : str
        Path to the train test folder.
    """
    print("Splitting the id data into train and test")
    if CV:
        from sklearn.model_selection import KFold, StratifiedKFold

        if stratify:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            train_test_folder = os.path.join(sample_folder, "CV_train_test", "strat")
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            train_test_folder = os.path.join(sample_folder, "CV_train_test", "no_strat")

        if not os.path.exists(train_test_folder):
            os.makedirs(train_test_folder)
        # kf.get_n_splits(X, y)
        # save train and test as pickle files
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            cv_folder = os.path.join(train_test_folder, "CV_" + str(i))
            train = [X[i] for i in train_index]
            test = [X[i] for i in test_index]
            if not os.path.exists(cv_folder):
                os.makedirs(cv_folder)
            with open(
                os.path.join(cv_folder, "train_" + str(i) + ".pickle"), "wb"
            ) as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(
                os.path.join(cv_folder, "test_" + str(i) + ".pickle"), "wb"
            ) as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        from sklearn.model_selection import train_test_split

        if stratify:
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X, y, test_size=test_size, stratify=y_strat, random_state=42
            )
            train_test_folder = os.path.join(sample_folder, "NO_CV_train_test", "strat")
        else:
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            train_test_folder = os.path.join(
                sample_folder, "NO_CV_train_test", "no_strat"
            )

        if not os.path.exists(train_test_folder):
            os.makedirs(train_test_folder)
        # save Xtrain and Xtest as pickle files
        with open(os.path.join(train_test_folder, "Xtrain.pickle"), "wb") as handle:
            pickle.dump(Xtrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_test_folder, "Xtest.pickle"), "wb") as handle:
            pickle.dump(Xtest, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save labels ytrain and ytest as pickle files
        with open(os.path.join(labels_folder_path, "ytrain.pickle"), "wb") as handle:
            pickle.dump(ytrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(labels_folder_path, "ytest.pickle"), "wb") as handle:
            pickle.dump(ytest, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_test_folder


# normalize the train and transform the test min max scaler
def omic_normalize(omic_train: pd.DataFrame, omic_test: pd.DataFrame) -> tuple:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    omic_train = scaler.fit_transform(omic_train)
    omic_test = scaler.transform(omic_test)
    return omic_train, omic_test


def train_test_save(
    j: int,
    df: pd.DataFrame,
    train: list,
    test: list,
    omic_normalize_dict: dict,
    data_folder_path: str,
    i: int,
    CV: bool = False,
    stratify: bool = True,
) -> pd.DataFrame:
    """
    Save the train and test data as csv files

    Parameters
    ----------

    j : int
        Omic number.
    df : pandas dataframe
        Dataframe containing the data.
    train : list
        List of training samples.
    test : list
        List of test samples.
    data_folder_path : str
        Path to the data folder.
    i : int
        In case oc CV, split number.
    CV : bool, optional
        Whether to use cross validation. The default is False.
    stratify : bool, optional
        Whether the stratified sampling has been used or not. The default is True.

    Returns
    -------
    omic_train : pandas dataframe
        Dataframe containing the training data.
    omic_test : pandas dataframe
        Dataframe containing the test data.
    """
    # split the data into train and test using the sample ids lists in train, test lists
    omic_train = df[df.index.isin(train)]
    # fill empty values with 0
    omic_train = omic_train.fillna(0)

    omic_test = df[df.index.isin(test)]
    # fill empty values with 0
    omic_test = omic_test.fillna(0)

    # normalize the data
    if omic_normalize_dict[j]:
        omic_train, omic_test = omic_normalize(omic_train, omic_test)
        # make it a dataframe like the original one
        omic_train = pd.DataFrame(omic_train, columns=df.columns)
        omic_test = pd.DataFrame(omic_test, columns=df.columns)

    if stratify:
        sample_type_path = os.path.join(data_folder_path, "strat")
    else:
        sample_type_path = os.path.join(data_folder_path, "no_strat")

    if CV:
        folder_name = "CV"
        folder_path = os.path.join(sample_type_path, folder_name)
        if not os.path.exists(folder_path):  # if the "CV" folder does not exist
            os.makedirs(folder_path)  # create the folder
        cv_folder = os.path.join(folder_path, "CV_" + str(i))
        if not os.path.exists(cv_folder):
            os.makedirs(cv_folder)
        train_file_name = str(j) + "_tr.csv"
        train_path = os.path.join(cv_folder, train_file_name)
        omic_train.to_csv(train_path, header=None, index=None)
        print("Training set saved as `", train_file_name, "in the folder:", folder_path)

        test_file_name = str(j) + "_te.csv"
        test_path = os.path.join(cv_folder, test_file_name)
        omic_test.to_csv(test_path, header=None, index=None)
        print("Test set saved as`", test_file_name, "in the folder:", folder_path)

    else:
        # save the train and test data as csv files
        folder_name = "NO_CV"
        folder_path = os.path.join(sample_type_path, folder_name)
        if not os.path.exists(folder_path):  # if the "NO_CV" folder does not exist
            os.makedirs(folder_path)  # create the folder

        train_file_name = str(j) + "_tr.csv"
        train_path = os.path.join(folder_path, train_file_name)
        omic_train.to_csv(train_path, header=None, index=None)
        print("Training set saved as `", train_file_name, "in the folder:", folder_path)

        test_file_name = str(j) + "_te.csv"
        test_path = os.path.join(folder_path, test_file_name)
        omic_test.to_csv(test_path, header=None, index=None)
        print("Test set saved as`", test_file_name, "in the folder:", folder_path)

    return omic_train, omic_test


# save labels from the labels dict
def save_labels(labels_dict, train, test, data_folder_path, i, CV=False):
    labels = pd.DataFrame.from_dict(labels_dict, orient="index")

    if CV:
        folder_name = "CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "CV" folder does not exist
            os.makedirs(folder_path)  # create the folder
        cv_folder = os.path.join(folder_path, "CV_" + str(i))
        if not os.path.exists(cv_folder):
            os.makedirs(cv_folder)
        label_train_path = os.path.join(cv_folder, "labels_tr.csv")
        label_test_path = os.path.join(cv_folder, "labels_te.csv")

        label_train = labels[labels.index.isin(train)]
        label_train.to_csv(label_train_path, header=None, index=None)
        print(
            "Labels for training set saved as `labels_tr.csv` in the folder:", cv_folder
        )
        label_test = labels[labels.index.isin(test)]
        label_test.to_csv(label_test_path, header=None, index=None)
        print("Label for test set saved as `labels_te.csv` in the folder:", cv_folder)

    else:
        folder_name = "NO_CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "NO_CV" folder does not exist
            os.makedirs(folder_path)  # create the folder

        label_train_path = os.path.join(folder_path, "labels_tr.csv")
        label_test_path = os.path.join(folder_path, "labels_te.csv")

        label_train = labels[labels.index.isin(train)]
        label_train.to_csv(label_train_path, header=None, index=None)
        print(
            "Labels for training set saved as `labels_tr.csv` in the folder:",
            label_train_path,
        )
        label_test = labels[labels.index.isin(test)]
        label_test.to_csv(label_test_path, header=None, index=None)
        print(
            "Label for test set saved as `labels_te.csv` in the folder:",
            label_test_path,
        )

    return


def save_sample_ids(sample_ids_dict: dict, sample_folder: str) -> print:
    """Save the sample ids as pickle file.

    Parameters
    ----------
    sample_ids_dict : dict
        Dictionary of sample ids.
    sample_folder : str
        Path to the sample folder.

    Returns
    -------
    print
        Print the path to the sample ids pickle file.
    """
    if not os.path.exists(sample_folder):  # if the sample folder does not exist
        os.makedirs(sample_folder)  # create the sample folder

    sample_ids_dict_save = os.path.join(sample_folder, "sample_ids_dict.pickle")

    if not os.path.exists(sample_ids_dict_save):
        # save sample_ids_dict dictionary as pickle file
        with open(
            sample_ids_dict_save, "wb"
        ) as handle:  # path to the sample_ids_dict pickle file
            pickle.dump(
                sample_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
            )  # save the sample_ids_dict pickle file

    return print("Sample ids saved as pickle file at: ", sample_folder)


def find_common_sample_ids(sample_ids_dict, sample_folder):
    print("Finding common sample ids")
    common_samples = os.path.join(sample_folder, "common_sample_ids.pickle")
    if not os.path.exists(common_samples):
        common_sample_ids = set.intersection(*map(set, sample_ids_dict.values()))
        with open(
            common_samples, "wb"
        ) as handle:  # path to the common_sample_ids pickle file
            pickle.dump(
                common_sample_ids, handle, protocol=pickle.HIGHEST_PROTOCOL
            )  # save the common_sample_ids pickle file
    return


def omicwise_filtering(
    i: int, main_data_folder_path: str, common_sample_ids: list
) -> None:
    """Filter the omic data based on the common sample ids

    Parameters
    ----------
    i : int
        Omic number.
    main_data_folder_path : str
        Path to the main data folder where omic folder is located.
    common_sample_ids : list
        List of common sample ids.

    Returns
    -------
    None
    """
    print("Filtering omic data for omic:", i)
    omic_path = os.path.join(main_data_folder_path, str(i))
    processed_data_file_name = (
        str(i) + "_processed_data.csv"
    )  # name of the processed data file
    df = pd.read_csv(
        os.path.join(omic_path, processed_data_file_name), index_col=0
    )  # read the processed data file
    df = df[df.index.isin(common_sample_ids)]  # keep the common samples
    common_processed_data_file_name = (
        str(i) + "_common_processed_data.csv"
    )  # name of the common processed data file
    df.to_csv(
        os.path.join(omic_path, common_processed_data_file_name)
    )  # save the common processed data file

    return


def dataset_summary(folder_name):
    print("Dataset:", folder_name)
    df_train = pd.read_csv("./" + folder_name + "/1_tr.csv", header=None)
    df_test = pd.read_csv("./" + folder_name + "/1_te.csv", header=None)
    print(
        "Number of features:",
        pd.read_csv("./" + folder_name + "/1_featname.csv", header=None).shape[0],
    )
    print("Total Number of samples:", df_train.shape[0] + df_test.shape[0])
    print(
        "Number of labels:",
        len(
            pd.read_csv("./" + folder_name + "/labels_te.csv", header=None)[0].unique()
        ),
    )
    print("-------------------------------------------------------------------")
    print(
        "Training set dimension:",
        df_train.shape,
    )
    print(
        "Number of samples for TRAINING:",
        df_train.shape[0],
    )
    print(
        "Number of labels for TRAINING:",
        pd.read_csv("./" + folder_name + "/labels_tr.csv", header=None).shape[0],
    )
    print("-------------------------------------------------------------------")
    print(
        "Test set dimension:",
        df_test.shape,
    )
    print(
        "Number of samples for TESTING:",
        df_test.shape[0],
    )
    print(
        "Number of labels for TESTING:",
        pd.read_csv("./" + folder_name + "/labels_te.csv", header=None).shape[0],
    )


def select_file_from_folder(folder_path: str) -> str:
    """
    Select the file from a folder which doest start with a number from 0-9

    Parameters
    ----------
    folder_path : str
        Path to the folder.

    Returns
    -------
    file_list[0] : str
        Name of the file.
    """
    file_list = os.listdir(folder_path)
    # if the file starts with a number from 0-9, remove it from the list
    for file in file_list:
        if file.startswith(tuple(map(str, range(10)))):
            file_list.remove(file)

    return file_list[0]


def find_numFolders_maxNumFolders(input: str) -> list:
    """
    Find the number of folders and the maximum number of folders

    Parameters
    ----------
    input : str
        Path to the folder.

    Returns
    -------
    intlistfolders : list
        List of folders.
    """
    intlistfolders = []
    list_subfolders_with_paths = [f.name for f in os.scandir(input) if f.is_dir()]
    for x in list_subfolders_with_paths:
        try:
            intval = int(x)
            # print(intval)
            intlistfolders.append(intval)
        except:
            pass
    return intlistfolders, max(intlistfolders)


# pipline for cml


def get_pipelines(options, DF, model):
    pl_preprocessor = build_preprocessor_pipeline(DF)

    # Build the entire pipeline
    pl = Pipeline(steps=[("preproc", pl_preprocessor), ("model", model)])

    return pl


def build_preprocessor_pipeline(DF: pd.DataFrame) -> Pipeline:
    """Build the preprocessor pipeline.

    Parameters
    ----------
    DF : pandas dataframe
        Dataframe containing the data.

    Returns
    -------
    pl_preprocessor : sklearn pipeline
        Preprocessor pipeline.
    """
    # numeric column names
    num_cols = DF.select_dtypes(exclude=["object"]).columns.tolist()

    # pipeline for numerical columns
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    pl_impute_encode = ColumnTransformer([("num", num_pipe, num_cols)])

    # full preprocessor pipeline
    pl_preprocessor = Pipeline([("impute_encode", pl_impute_encode)])

    return pl_preprocessor


def get_expname_datetime(options: dict) -> str:
    """Get the experiment name with date and time.

    Parameters
    ----------
    options : dict
        Dictionary of options.

    Returns
    -------
    expname : str
        Experiment name.
    """
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    expname = options["model"] + "_" + options["mode"] + "_" + dt_string
    print("exp. name =" + expname)
    return expname


# print epoch loss
def print_epoch_loss(epoch, train_loss, test_loss):
    print("Epoch: %d, train loss: %f, test loss: %f" % (epoch, train_loss, test_loss))
    return


# key as x axis and values which are dictionaries as y axis
def plot_epoch_loss(epoch_loss_dict: dict, fig_path: str) -> None:
    """
    Plot the epoch loss

    Parameters
    ----------
    epoch_loss_dict : dict
        Dictionary of epoch loss.
    fig_path : str
        Path to save the figure.

    Returns
    -------
    None
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import os

    # convert epoch loss dictionary to dataframe
    df = pd.DataFrame.from_dict(epoch_loss_dict)
    df = df.transpose()
    df = df.reset_index()
    df = df.rename(columns={"index": "epoch"})
    df = df.melt("epoch", var_name="cols", value_name="vals")
    # plot epoch loss
    plt.figure()
    ax = sns.lineplot(x="epoch", y="vals", hue="cols", data=df)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.savefig(fig_path)
    plt.show()

    return


def plot_threshold_curve(y, yproba, title, figsize, figpath=None):
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y, yproba > t) for t in thresholds]

    plt.plot(thresholds, f1_scores, label="F1 Score vs. Threshold Curve")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.savefig(figpath)


def result_plot(
    labels: np.ndarray,
    prob: np.ndarray,
    latest_model: str,
    cm_normalised: bool = False,
) -> None:
    """
    Plot the results

    Parameters
    ----------
    labels : numpy array
        Labels.
    prob : numpy array
        Probabilities.
    latest_model : str
        Path to the latest model.
    cm_normalised : bool, optional
        Whether to normalise the confusion matrix. The default is False.

    Returns
    -------
    None
    """
    from sklearn.metrics import (
        f1_score,
        matthews_corrcoef,
        roc_curve,
        roc_auc_score,
        precision_recall_curve,
        auc,
        average_precision_score,
    )

    if not os.path.exists(latest_model):
        os.makedirs(latest_model)
    model_name = os.path.basename(os.path.normpath(latest_model))
    import scikitplot as skplt

    y = labels
    # yhat = yyhat_dict[key]["yhat"]
    yproba = prob
    # save each figure seprpately

    # F1 Score vs. Threshold Curve
    # save plot
    image_name = "mogonet_" + model_name + "_f1_scores_vs_threshold.png"
    figpath = os.path.join(latest_model, image_name)
    plot_threshold_curve(
        y,
        yproba[:, 1],
        title="F1 Score vs. Threshold Curve" + model_name,
        figsize=(8, 8),
        figpath=figpath,
    )

    # save the above given path

    skplt.metrics.plot_confusion_matrix(
        y,
        yproba.argmax(1),
        normalize=cm_normalised,
        title="Confusion Matrix: " + model_name,
        figsize=(8, 8),
    )
    # save plot
    image_name = "mogonet_" + model_name + "_cm.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_roc_curve(y, yproba, title="ROC Curve: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_roc_curve.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_roc(y, yproba, title="ROC Plot: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_roc.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_precision_recall_curve(
        y, yproba, title="PR Curve2: " + model_name
    )
    # save plot
    image_name = "mogonet_" + model_name + "_pr_curve2.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_precision_recall(y, yproba, title="PR Curve: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_pr.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_cumulative_gain(
        y, yproba, title="Cumulative Gains Chart: " + model_name
    )
    # save plot
    image_name = "mogonet_" + model_name + "_gain.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_lift_curve(y, yproba, title="Lift Curve: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_lift.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    print(f"Saved figure: {figpath}")

    return


# # Matthews Correlation Coefficient
# mcc = matthews_corrcoef(y_test, clf.predict(X_test))
# print(f"Matthews Correlation Coefficient: {mcc}")


def preprocessing_omic_data(omics_list: list, main_data_folder_path: str) -> dict:
    """Store the sample ids in a dictionary

    Parameters
    ----------
    omics_list : list
        List of omics.
    main_data_folder_path : str
        Path to the main data folder where omic folder is located.

    Returns
    -------
    sample_folder : str
        Path to the sample folder.
    sample_ids_dict : dict
        Dictionary of sample ids.
    """
    # dictionary to store the sample ids
    sample_ids_dict = {}
    for i in omics_list:
        omic_path = os.path.join(
            main_data_folder_path, str(i)
        )  # path to the omic folder

        processed_data_file_name = str(i) + "_processed_data.csv"
        processed_file_path = os.path.join(omic_path, processed_data_file_name)
        check_if_it_is_needed_to_processed = os.path.exists(processed_file_path)
        if not check_if_it_is_needed_to_processed:
            print(f"Processing the omic number {i} data")
            for f in os.scandir(
                omic_path
            ):  # iterate through the files in the omic folder
                if f.is_file():  # if the file is a file
                    raw_data_file = f.name  # get the name of the file
                raw_data_file_path = os.path.join(
                    omic_path, raw_data_file
                )  # path to the raw data file
                df = pd.read_csv(raw_data_file_path, sep=",")
                # make a index into a column
                # df.reset_index(inplace=True)
                df = df.T
                df.columns = df.iloc[0:1].values[0]
                df = df.iloc[1:, :]
                # fill empty values with 0
                df = df.fillna(0)
                # save the processed data file
                df.to_csv(processed_file_path)  # save the processed data file
                sample_ids_dict[i] = df.index  # save the sample ids
        else:
            df = pd.read_csv(processed_file_path, index_col=0)
            sample_ids_dict[i] = df.index

    return sample_ids_dict


def differential_ananlysis_core(
    omic_train: pd.DataFrame,
    strat_dict: dict,
    filter_wrt: str,
    method: str,
) -> pd.DataFrame:
    """
    Perform differential analysis core

    Parameters
    ----------
    omic_train : pandas dataframe
        Dataframe containing the training data.
    control_group : pandas dataframe
        Dataframe containing the control group.
    case_group : pandas dataframe
        Dataframe containing the case group.
    da_wrt : str
        Differential analysis with respect to.

    Returns
    -------
    differential_results_df : pandas dataframe
        Dataframe containing the differential analysis results.
    """
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import f_oneway, ttest_ind
    from sklearn.feature_selection import f_classif

    # add column to omic_train using labels_dict and index as the keys
    # omic_train["fetal_near_miss"] = omic_train.index.map(labels_dict)

    # # Define your control and disease groups
    # control_group = omic_train[omic_train["fetal_near_miss"] == "0"]
    # case_group = omic_train[omic_train["fetal_near_miss"] == "1"]

    # Drop the 'fetal_near_miss' column from the omic_train DataFrame
    # omic_train.drop("fetal_near_miss", axis=1, inplace=True)

    # List to store results
    # differential_results = []

    # Perform differential analysis for each omic feature
    # for col_name in omic_train.columns:
    #     # t-test
    #     t_stat, p_value_t = ttest_ind(control_group[col_name], case_group[col_name])
    #     # ANOVA F test
    #     # f_stat_oneway, p_value_f_oneway = f_oneway(
    #     #     control_group[col_name], case_group[col_name]
    #     # )

    #     # # find log_fold_change for each feature
    #     # # Calculate the mean of the case and control groups
    #     # mean_case = case_group[col_name].mean()
    #     # mean_control = control_group[col_name].mean()

    #     # # Avoid division by zero by checking if mean_control is zero
    #     # if mean_control == 0:
    #     #     fold_change = float("inf")  # Set fold change to infinity
    #     # else:
    #     #     fold_change = mean_case / mean_control
    #     # # Perform log transformation
    #     # log_fold_change = np.log(fold_change)

    #     # Store the results
    #     differential_results.append(
    #         {
    #             "feature": col_name,
    #             "t-stat": t_stat,
    #             "p-value(t-test)": p_value_t,
    #             # "f-stat_oneway": f_stat_oneway,
    #             # "p-value(f-test)": p_value_f_oneway,
    #             # "fold_change": log_fold_change,
    #         }
    #     )
    # # Convert the results to a DataFrame
    # differential_results_df = pd.DataFrame(differential_results)

    F, p_value_f = f_classif(omic_train, omic_train.index.map(strat_dict))
    # Add a column f-stat, p-value(f-test) to the DataFrame
    differential_results_df = pd.DataFrame(
        {"feature": omic_train.columns, "f-stat": F, "p-value(f-test)": p_value_f}
    )
    # differential_results_df["f-stat"] = F
    # differential_results_df["p-value(f-test)"] = p_value_f

    # Adjust p-values for multiple testing (e.g., using the Benjamini-Hochberg procedure - fdr_bh or Bonferroni as Bonferroni)
    # differential_results_df["adjusted_p-value(t-test)"] = multipletests(
    #     differential_results_df["p-value(t-test)"], method=method
    # )[1]
    differential_results_df["adjusted_p-value(f-test)"] = multipletests(
        differential_results_df["p-value(f-test)"], method=method
    )[1]

    # Sort the results
    differential_results_df.sort_values(by=filter_wrt, ascending=True, inplace=True)

    return differential_results_df


def differential_analysis(
    j: int,
    omic_path: str,
    common_df: pd.DataFrame,
    train: list,
    strat_dict: dict,
    filter_wrt: str,
    method: str,
    da_threshold: float,
    i: int,
    CV: bool,
    stratify: bool,
) -> pd.DataFrame:
    """
    Perform differential analysis

    Parameters
    ----------
    j : int
        Omic number.
    omic_path : str
        Path to the omic folder.
    common_df : pandas dataframe
        Dataframe containing the common data.
    train : list
        List of training samples.
    labels_dict : dict
        Dictionary of labels.
    filter_wrt : str
        Differential analysis with respect to.
    method : str
        Method for multiple testing correction.
    da_threshold : float
        Differential analysis threshold.
    i : int
        In case of CV, split number.
    CV : bool, optional
        Whether to use cross validation. The default is False.
    stratify : bool, optional
        Whether to use stratified sampling. The default is True.

    Returns
    -------
    differential_results_df : pandas dataframe
        Dataframe containing the differential analysis results.
    """

    if stratify:
        omic_path_sample_type = os.path.join(omic_path, "strat")
    else:
        omic_path_sample_type = os.path.join(omic_path, "no_strat")
    if CV:
        # create file name as per the omic number and cv split number
        diff_result_fiel_name = (
            str(j)
            + "_differential_results_"
            + method
            + "_"
            + filter_wrt
            + "_"
            + str(da_threshold)
            + "_CV_"
            + str(i)
            + ".csv"
        )
        diff_folder = os.path.join(omic_path_sample_type, "differential_analysis", "CV")
        # if path does not exist, create the path
    else:
        # create file name as per the omic number
        diff_result_fiel_name = (
            str(j)
            + "_differential_results_"
            + method
            + "_"
            + filter_wrt
            + "_"
            + str(da_threshold)
            + ".csv"
        )
        diff_folder = os.path.join(
            omic_path_sample_type, "differential_analysis", "NO_CV"
        )
        # if path does not exist, create the path

    if not os.path.exists(diff_folder):
        os.makedirs(diff_folder)
    differential_results_file_path = os.path.join(diff_folder, diff_result_fiel_name)

    # filter the common_df using the train list
    omic_train = common_df[common_df.index.isin(train)]

    differential_results_df = differential_ananlysis_core(
        omic_train=omic_train,
        strat_dict=strat_dict,
        filter_wrt=filter_wrt,
        method=method,
    )

    # Save the results
    differential_results_df.to_csv(differential_results_file_path)

    return differential_results_df, diff_folder


# select the column da_wrt and filter by da_threshold
def filter_differential_results(
    j: int,
    diff_folder: str,
    differential_results_df: pd.DataFrame,
    filter_wrt: str,
    method: str,
    da_threshold: float,
    i: int = None,
    CV: bool = False,
) -> pd.DataFrame:
    """
    Select the column da_wrt and filter by da_threshold

    Parameters
    ----------
    j : int
        Omic number.
    diff_folder : str
        Path to the differential analysis folder.
    differential_results_df : pandas dataframe
        Differential analysis results dataframe.
    da_wrt : str
        Differential analysis with respect to.
    da_threshold : float
        Differential analysis threshold.
    i : int
        In case of CV, split number.
    CV : bool, optional
        Whether to use cross validation. The default is False.

    Returns
    -------
    filter_differential_results_df : pandas dataframe
        Filtered differential analysis results dataframe.
    """
    if CV:
        # create file name as per the omic number and cv split number
        filter_diff_result_file_name = (
            str(j)
            + "_filter_differential_results_"
            + method
            + "_"
            + filter_wrt
            + "_"
            + str(da_threshold)
            + "_CV_"
            + str(i)
            + ".csv"
        )
        # diff_folder_path = diff_folder
        diff_folder_path = os.path.join(
            diff_folder, "filtered"
        )  # seprate folder for filtered results
        # if path does not exist, create the path
        if not os.path.exists(diff_folder_path):
            os.makedirs(diff_folder_path)
        filter_differential_results_file_path = os.path.join(
            diff_folder_path, filter_diff_result_file_name
        )
    else:
        # create file name as per the omic number
        filter_diff_result_file_name = (
            str(j)
            + "_filter_differential_results_"
            + method
            + "_"
            + filter_wrt
            + "_"
            + str(da_threshold)
            + ".csv"
        )
        diff_folder_path = os.path.join(
            diff_folder, "filtered"
        )  # seprate folder for filtered results
        # if path does not exist, create the path
        if not os.path.exists(diff_folder_path):
            os.makedirs(diff_folder_path)
        filter_differential_results_file_path = os.path.join(
            diff_folder_path, filter_diff_result_file_name
        )

    # filter the differential results dataframe
    filter_differential_results_df = differential_results_df[
        differential_results_df[filter_wrt] <= da_threshold
    ]

    # save the filtered differential results dataframe
    filter_differential_results_df.to_csv(
        filter_differential_results_file_path, index=False
    )

    return filter_differential_results_df


def score_plot_new(path, info, save_path):
    from matplotlib import legend
    import matplotlib.pyplot as plt
    import seaborn as sns

    df1 = pd.read_csv(path, index_col=0)
    # drop "CM"
    df1 = df1.drop(columns=["CM"])
    # print(df1)
    # df = df1.groupby("model").agg(["mean", "std"])
    # df = df1.T
    # df = df.reset_index()

    # df = df.rename(columns={"level_0": "metric", "level_1": "mean/std"})
    df = df1.melt(
        id_vars="model",
        value_vars=["ACC", "AUC", "F1"],
        var_name="metric",
        value_name="score",
    )

    df["model"] = df["model"].replace(
        {
            "XGBC": "XGBoost",
            "RFC": "Random Forest",
            "SVC": "Support Vector Machine",
            "DTC": "Decision Tree",
            "LRC": "Logistic Regression",
            "ElasticNet": "Elastic Net",
            "mogonet": "MOGONET",
        }
    )

    # print(df)
    # # print(df)
    # #plot metric and mean/std
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    # palletr dict for the model, light colours
    palette = {
        "XGBoost": "red",
        "Random Forest": "blue",
        "Support Vector Machine": "purple",
        "Decision Tree": "orange",
        "Logistic Regression": "yellow",
        "Elastic Net": "pink",
        "MOGONET": "green",
    }
    # keep the oder of the model fixed

    g = sns.barplot(
        x="metric",
        y="score",
        hue="model",
        hue_order=[
            "Support Vector Machine",
            "Logistic Regression",
            "Elastic Net",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "MOGONET",
        ],
        estimator="mean",
        data=df,
        palette=palette,
        # palette="tab10",  # pallet colour full list: https://seaborn.pydata.org/tutorial/color_palettes.html
        errorbar=("pi", 75),  # "sd",  # ("pi", 50),#
        capsize=0.05,
        alpha=0.8,
        errcolor=".4",
        linewidth=1,
        # edgecolor=".2",
    )

    sns.swarmplot(
        x="metric",
        y="score",
        hue="model",
        hue_order=[
            "Support Vector Machine",
            "Logistic Regression",
            "Elastic Net",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "MOGONET",
        ],
        data=df,
        color="0",
        alpha=0.35,
        dodge=True,
        legend=False,
    )
    # # sns.set_theme(style="whitegrid")
    # # g = sns.catplot(
    # #     x="metric", y="score", hue="model", data=df, kind="bar", legend=False,
    # # )
    # # sns edit legend
    # # sns.move_legend(g, "upper left",loc=1)
    # # add title
    # plt.title()
    g.set_title("Model performace for the " + info)
    g.set_xlabel("Mertic")
    g.set_ylabel("Score")

    # change legend title
    plt.legend(title="Models", bbox_to_anchor=(1, 0.7))

    save_as = info + ".png"
    save_path = os.path.join(save_path, save_as)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
