import torch                                            # PyTorch to create and apply deep learning models
from torch import nn, optim                             # nn for neural network layers and optim for training optimizers
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd                                     # Pandas to handle the data in dataframes
from datetime import datetime                           # datetime to use proper date and time formats
import os                                               # os handles directory/workspace changes
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm import tqdm                                   # tqdm allows to track code execution progress
from tqdm import tqdm_notebook                          # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric
from NeuralNetwork import NeuralNetwork                 # Import the neural network model class
from sklearn.metrics import roc_auc_score               # ROC AUC model performance metric
import warnings                                         # Print warnings for bad practices
import sys                                              # Identify types of exceptions

# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
random_seed = 0

# Set random seed to the specified value
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Exceptions

class ColumnNotFoundError(Exception):
   """Raised when the column name is not found in the dataframe."""
   pass


# Auxiliary functions

def dataframe_missing_values(df, column=None):
    '''Returns a dataframe with the percentages of missing values of every column
    of the original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe which the user wants to analyze for missing values.
    column : string, default None
        Optional argument which, if provided, makes the function only return
        the percentage of missing values in the specified column.

    Returns
    -------
    missing_value_df : pandas.DataFrame
        DataFrame containing the percentages of missing values for each column.
    col_percent_missing : float
        If the "column" argument is provided, the function only returns a float
        corresponfing to the percentage of missing values in the specified column.
    '''
    if column is None:
        columns = df.columns
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': columns,
                                         'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        return missing_value_df
    else:
        col_percent_missing = df[column].isnull().sum() * 100 / len(df)
        return col_percent_missing


def get_clean_label(orig_label, clean_labels, column_name=None):
    '''Gets the clean version of a given label.

    Parameters
    ----------
    orig_label : string
        Original label name that needs to be converted to the new format.
    clean_labels : dict
        Dictionary that converts each original label into a new, cleaner designation.
    column_name : string, default None
        Optional parameter to indicate a column name, which is used to specify better the
        missing values.

    Returns
    -------
    key : string
        Returns the dictionary key from clean_labels that corresponds to the translation
        given to the input label orig_label.
    '''
    for key in clean_labels:
        if orig_label in clean_labels[key]:
            return key

    # Remaining labels (or lack of one) are considered as missing data
    if column_name is not None:
        return f'{column_name}_missing_value'
    else:
        return 'missing_value'


def is_one_hot_encoded_column(df, column):
    '''Checks if a given column is one hot encoded.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that will be used, which contains the specified column.
    column : string
        Name of the column that will be checked for one hot encoding.

    Returns
    -------
    bool
        Returns true if the column is in one hot encoding format.
        Otherwise, returns false.
    '''
    # Check if it only has 2 possible values
    if df[column].nunique() == 2:
        # Check if the possible values are all numeric
        if all([isinstance(x, numbers.Number) for x in df[column].unique()]):
            # Check if the only possible values are 0 and 1 (and ignore NaN's)
            if (np.sort(list(set(np.nan_to_num(df[column].unique())))) == [0, 1]).all():
                return True
    return False


def list_one_hot_encoded_columns(df):
    '''Lists the columns in a dataframe which are in a one hot encoding format.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that will be used checked for one hot encoded columns.

    Returns
    -------
    list of strings
        Returns a list of the column names which correspond to one hot encoded columns.
    '''
    return [col for col in df.columns if is_one_hot_encoded_column(df, col)]


def one_hot_encoding_dataframe(df, columns, std_name=True, has_nan=False, join_rows=True, join_by=['subject_id', 'ts']):
    '''Transforms a specified column from a dataframe into a one hot encoding representation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that will be used, which contains the specified column.
    columns : list of strings
        Name of the column(s) that will be conveted to one hot encoding. Even if it's just one
        column, please provide inside a list.
    std_name : bool, default True
        If set to true, changes the name of the categorical values into lower case, with words
        separated by an underscore instead of space.
    has_nan : bool, default False
        If set to true, will first fill the missing values (NaN) with the string
        f'{column}_missing_value'.
    join_rows : bool, default True
        If set to true, will group the rows created by the one hot encoding by summing the
        boolean values in the rows that have the same identifiers.
    join_by : string or list, default ['subject_id', 'ts'])
        Name of the column (or columns) which serves as a unique identifier of the dataframe's
        rows, which will be used in the groupby operation if the parameter join_rows is set to
        true. Can be a string (single column) or a list of strings (multiple columns).

    Raises
    ------
    ColumnNotFoundError
        Column name not found in the dataframe.

    Returns
    -------
    ohe_df : pandas.Dataframe
        Returns a new dataframe with the specified column in a one hot encoding representation.
    '''
    for col in columns:
        if has_nan:
            # Fill NaN with "missing_value" name
            df[col].fillna(value='missing_value', inplace=True)

        # Check if the column exists
        if col not in df:
            raise ColumnNotFoundError('Column name not found in the dataframe.')

        if std_name:
            # Change categorical values to only have lower case letters and underscores
            df[col] = df[col].apply(lambda x: str(x).lower().replace(' ', '_').replace(',', '_and'))

        # Cast the variable into the built in pandas Categorical data type
        df[col] = pd.Categorical(df[col])

    # Apply the one hot encoding to the specified columns
    ohe_df = pd.get_dummies(df, columns=columns)

    if join_rows:
        # Columns which are one hot encoded
        ohe_columns = list_one_hot_encoded_columns(ohe_df)

        # Group the rows that have the same identifiers
        ohe_df = ohe_df.groupby(join_by).sum(min_count=1).reset_index()

        # Clip the one hot encoded columns to a maximum value of 1
        # (there might be duplicates which cause values bigger than 1)
        ohe_df.loc[:, ohe_columns] = ohe_df[ohe_columns].clip(upper=1)

    return ohe_df


def is_definitely_string(x):
    '''Reports if a value is actually a real string or if it has some number in it.

    Parameters
    ----------
    x
        Any value which will be judged to be either a real string or numeric.

    Returns
    -------
    boolean
        Returns a boolean, being it True if it really is a string or False if it's
        either numeric data or a string with a number inside.
    '''
    if isinstance(x, int) or isinstance(x, float):
        return False

    try:
        float(x)
        return False

    except:
        return isinstance(x, str)


def remove_rows_unmatched_key(df, key, columns):
    '''Remove rows corresponding to the keys that weren't in the dataframe merged at the right.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe resulting from a asof merge which will be searched for missing values.
    key : string
        Name of the column which was used as the "by" key in the asof merge. Typically
        represents a temporal feature from a time series, such as days or timestamps.
    columns : list of strings
        Name of the column(s), originating from the dataframe which was merged at the
        right, which should not have any missing values. If it has, it means that
        the corresponding key wasn't present in the original dataframe. Even if there's
        just one column to analyze, it should be received in list format.

    Returns
    -------
    df : pandas.DataFrame
        Returns the input dataframe but without the rows which didn't have any values
        in the right dataframe's features.
    '''
    for k in tqdm_notebook(df[key].unique()):
        # Variable that count the number of columns which don't have any value
        # (i.e. all rows are missing values) for a given identifier 'k'
        num_empty_columns = 0

        for col in columns:
            if df[df[key] == k][col].isnull().sum() == len(df[df[key] == k]):
                # Found one more column which is full of missing values for identifier 'k'
                num_empty_columns += 1

        if num_empty_columns == len(columns):
            # Eliminate all rows corresponding to the analysed key if all the columns
            # are empty for the identifier 'k'
            df = df[~(df[key] == k)]

    return df


def dataframe_to_padded_tensor(df, seq_len_dict, n_ids, n_inputs, id_column='subject_id', data_type='PyTorch', padding_value=999999):
    '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

    Parameters
    ----------
    df : pandas.Dataframe
        Data in a Pandas dataframe format which will be padded and converted
        to the requested data type.
    seq_len_dict : dictionary
        Dictionary containing the original sequence lengths of the dataframe.
    n_ids : int
        Total number of subject identifiers in a dataframe.
        Example: Total number of patients in a health dataset.
    n_inputs : int
        Total number of input features present in the dataframe.
    id_column : string, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    data_type : string, default 'PyTorch'
        Indication of what kind of output data type is desired. In case it's
        set as 'NumPy', the function outputs a NumPy array. If it's 'PyTorch',
        the function outputs a PyTorch tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    arr : torch.Tensor or numpy.array
        PyTorch tensor or NumPy array version of the dataframe, after being
        padded with the specified padding value to have a fixed sequence
        length.
    '''
    # Max sequence length (e.g. patient with the most temporal events)
    max_seq_len = seq_len_dict[max(seq_len_dict, key=seq_len_dict.get)]

    # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
    arr = np.ones((n_ids, max_seq_len, n_inputs)) * padding_value

    # Iterator that outputs each unique identifier (e.g. each patient in the dataset)
    id_iter = iter(df[id_column].unique())

    # Count the iterations of ids
    count = 0

    # Assign each value from the dataframe to the numpy array
    for idt in id_iter:
        arr[count, :seq_len_dict[idt], :] = df[df[id_column] == idt].to_numpy()
        arr[count, seq_len_dict[idt]:, :] = padding_value
        count += 1

    # Make sure that the data type asked for is a string
    if not isinstance(data_type, str):
        raise Exception('ERROR: Please provide the desirable data type in a string format.')

    if data_type.lower() == 'numpy':
        return arr
    elif data_type.lower() == 'pytorch':
        return torch.from_numpy(arr)
    else:
        raise Exception('ERROR: Unavailable data type. Please choose either NumPy or PyTorch.')


def normalize_data(df, data=None, id_columns=['subject_id', 'ts'], normalization_method='z-score',
                   columns_to_normalize=None, see_progress=True):
    '''Performs data normalization to a continuous valued tensor or dataframe,
       changing the scale of the data.

    Parameters
    ----------
    df : pandas.Dataframe
        Original pandas dataframe which is used to correctly calculate the
        necessary statistical values used in the normalization. These values
        can't be calculated from the tensor as it might have been padded. If
        the data tensor isn't specified, the normalization is applied directly
        on the dataframe.
    data : torch.Tensor, default None
        PyTorch tensor corresponding to the data which will be normalized
        by the specified normalization method. If the data tensor isn't
        specified, the normalization is applied directly on the dataframe.
    id_columns : list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be normalized.
    normalization_method : string, default 'z-score'
        Specifies the normalization method used. It can be a z-score
        normalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a
        min-max normalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_normalize : list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be normalized. Otherwise, all continuous columns will be normalized.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.

    Returns
    -------
    data : pandas.Dataframe or torch.Tensor
        Normalized Pandas dataframe or PyTorch tensor.
    '''
    # Check if specific columns have been specified for normalization
    if columns_to_normalize is None:
        # List of binary or one hot encoded columns
        binary_cols = list_one_hot_encoded_columns(df)

        # Normalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_normalize = [col for col in df.columns if col not in binary_cols and col not in id_columns]

    if type(normalization_method) is not str:
        raise ValueError('Argument normalization_method should be a string. Available options \
                         are \'z-score\' and \'min-max\'.')

    if normalization_method.lower() == 'z-score':
        column_means = dict(df[columns_to_normalize].mean())
        column_stds = dict(df[columns_to_normalize].std())

        # Check if the data being normalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being normalized
            data = df.copy()

            # Normalize the right columns
            for col in iterations_loop(columns_to_normalize, see_progress=see_progress):
                data[col] = (data[col] - column_means[col]) / column_stds[col]

        # Otherwise, the tensor is normalized
        else:
            # Dictionary to convert the the tensor's column indeces into the dataframe's column names
            idx_to_name = dict(enumerate(df.columns))

            # Dictionary to convert the dataframe's column names into the tensor's column indeces
            name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

            # List of indeces of the tensor's columns which are needing normalization
            tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]

            # Normalize the right columns
            for col in iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                data[:, :, col] = (data[:, :, col] - column_means[idx_to_name[col]]) / column_stds[idx_to_name[col]]

    elif normalization_method.lower() == 'min-max':
        column_mins = dict(df[columns_to_normalize].min())
        column_maxs = dict(df[columns_to_normalize].max())

        # Check if the data being normalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being normalized
            data = df.copy()

            # Normalize the right columns
            for col in iterations_loop(columns_to_normalize, see_progress=see_progress):
                data[col] = (data[col] - column_mins[col]) / (column_maxs[col] - column_mins[col])

        # Otherwise, the tensor is normalized
        else:
            # Dictionary to convert the the tensor's column indeces into the dataframe's column names
            idx_to_name = dict(enumerate(df.columns))

            # Dictionary to convert the dataframe's column names into the tensor's column indeces
            name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

            # List of indeces of the tensor's columns which are needing normalization
            tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]

            # Normalize the right columns
            for col in iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                data[:, :, col] = (data[:, :, col] - column_mins[idx_to_name[col]]) / \
                                  (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]])

    else:
        raise ValueError(f'{normalization_method} isn\'t a valid normalization method. Available options \
                         are \'z-score\' and \'min-max\'.')

    return data


def denormalize_data(df, data, id_columns=['subject_id', 'ts'], normalization_method='z-score',
                     columns_to_denormalize=None, see_progress=True):
    '''Performs data denormalization to a continuous valued tensor or dataframe,
       changing the scale of the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Original pandas dataframe which is used to correctly calculate the
        necessary statistical values used in the denormalization. These values
        can't be calculated from the tensor as it might have been padded.
    data : torch.Tensor or pandas.DataFrame
        PyTorch tensor or pandas dataframe corresponding to the data which will
        be denormalized by the specified normalization method.
    id_columns : list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be denormalized.
    normalization_method : string, default 'z-score'
        Specifies the normalization method used. It can be a z-score
        normalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a
        min-max normalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_denormalize : list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be denormalized. Otherwise, all non identifier continuous columns
        will be denormalized.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.

    Returns
    -------
    data : pandas.DataFrame or torch.Tensor
        Denormalized Pandas dataframe or PyTorch tensor.
    '''
    # Variable that will store the denormalized data
    denorm_data = data.clone().detach()

    # Check if specific columns have been specified for denormalization
    if columns_to_denormalize is None:
        # List of binary or one hot encoded columns
        binary_cols = list_one_hot_encoded_columns(df)

        # Denormalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_denormalize = [col for col in df.columns if col not in binary_cols and col not in id_columns]

    if type(normalization_method) is not str:
        raise ValueError('Argument normalization_method should be a string. Available options \
                         are \'z-score\' and \'min-max\'.')

    if normalization_method.lower() == 'z-score':
        column_means = dict(df[columns_to_denormalize].mean())
        column_stds = dict(df[columns_to_denormalize].std())

        # Check if the data being denormalized is a dataframe
        if type(data) is pd.DataFrame:
            # Denormalize the right columns
            for col in iterations_loop(columns_to_denormalize, see_progress=see_progress):
                denorm_data[col] = data[col] * column_stds[col] + column_means[col]

        # Otherwise, the tensor is denormalized
        else:
            # Dictionary to convert the the tensor's column indeces into the dataframe's column names
            idx_to_name = dict(enumerate(df.columns))

            # Dictionary to convert the dataframe's column names into the tensor's column indeces
            name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

            # List of indeces of the tensor's columns which are needing denormalization
            tensor_columns_to_denormalize = [name_to_idx[name] for name in columns_to_denormalize]

            # Denormalize the right columns
            for col in iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
                denorm_data[:, :, col] = data[:, :, col] * column_stds[idx_to_name[col]] + column_means[idx_to_name[col]]

    elif normalization_method.lower() == 'min-max':
        column_mins = dict(df[columns_to_denormalize].min())
        column_maxs = dict(df[columns_to_denormalize].max())

        # Check if the data being normalized is directly the dataframe
        if type(data) is pd.DataFrame:
            # Denormalize the right columns
            for col in iterations_loop(columns_to_denormalize, see_progress=see_progress):
                denorm_data[col] = data[col] * (column_maxs[col] - column_mins[col]) + column_mins[col]

        # Otherwise, the tensor is denormalized
        else:
            # Dictionary to convert the the tensor's column indeces into the dataframe's column names
            idx_to_name = dict(enumerate(df.columns))

            # Dictionary to convert the dataframe's column names into the tensor's column indeces
            name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

            # List of indeces of the tensor's columns which are needing normalization
            tensor_columns_to_denormalize = [name_to_idx[name] for name in columns_to_normalize]

            # Denormalize the right columns
            for col in iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
                denorm_data[:, :, col] = data[:, :, col] * (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]]) \
                                         + column_mins[idx_to_name[col]]

    else:
        raise ValueError(f'{normalization_method} isn\'t a valid normalization method. Available options \
                         are \'z-score\' and \'min-max\'.')

    return denorm_data


def missing_values_imputation(tensor):
    '''Performs missing values imputation to a tensor corresponding to a single column.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor corresponding to a single column which will be imputed.

    Returns
    -------
    tensor : torch.Tensor
        Imputed PyTorch tensor.
    '''
    # Replace NaN's with zeros
    tensor = torch.where(tensor != tensor, torch.zeros_like(tensor), tensor)

    return tensor


def create_train_sets(dataset, validation_ratio=0.1, batch_size=32,
                      get_indeces=True, random_seed=42, shuffle_dataset=True):
    '''Distributes the data into train, validation and test sets and returns the respective data loaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object which will be used to train, validate and test the model.
    validation_ratio : float, default 0.1
        Number from 0 to 1 which indicates the percentage of the data
        from the training set which is used for validation purposes.
        A value of 0.0 corresponds to not using validation.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    get_indeces : bool, default True
        If set to True, the function returns the dataloader objects of
        the train, validation and test sets and also the indices of the
        sets' data. Otherwise, it only returns the data loaders.
    random_seed : int, default 42
        Seed used when shuffling the data.
    shuffle_dataset : bool, default True
        If set to True, the data of which set is shuffled.

    Returns
    -------
    train_data : torch.Tensor
        Data which will be used during training.
    val_data : torch.Tensor
        Data which will be used to evaluate the model's performance
        on a validation set during training.
    '''
    # Create data indices for training and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(validation_ratio * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[val_split:], indices[:val_split]

    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create dataloaders for each set, which will allow loading batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    if get_indeces:
        # Return the data loaders and the indices of the sets
        return train_dataloader, val_dataloader, train_indices, val_indices
    else:
        # Just return the data loaders of each set
        return train_dataloader, val_dataloader


def load_checkpoint(filepath):
    '''Load a model from a specified path and name.

    Parameters
    ----------
    filepath : str
        Path to the model being loaded, including it's own file name.

    Returns
    -------
    model : nn.Module
        The loaded model with saved weight values.
    '''
    checkpoint = torch.load(filepath)
    model = NeuralNetwork(checkpoint['n_inputs'],
                          checkpoint['n_hidden'],
                          checkpoint['n_outputs'],
                          checkpoint['n_layers'],
                          checkpoint['p_dropout'])
    model.load_state_dict(checkpoint['state_dict'])

    return model


def sort_by_seq_len(data, seq_len_dict, labels=None, id_column=0):
    '''Sort the data by sequence length in order to correctly apply it to a
    PyTorch neural network.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor on which sorting by sequence length will be applied.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    labels : torch.Tensor, default None
        Labels corresponding to the data used, either specified in the input
        or all the data that the interpreter has.
    id_column : int, default 0
        Number of the column which corresponds to the subject identifier in
        the data tensor.

    Returns
    -------
    sorted_data : torch.Tensor, default None
        Data tensor already sorted by sequence length.
    sorted_labels : torch.Tensor, default None
        Labels tensor already sorted by sequence length. Only outputed if the
        labels data is specified in the input.
    x_lengths : list of int
        Sorted list of sequence lengths, relative to the input data.
    '''
    # Get the original lengths of the sequences, for the input data
    x_lengths = [seq_len_dict[id] for id in list(data[:, 0, id_column].numpy())]

    is_sorted = all(x_lengths[i] >= x_lengths[i+1] for i in range(len(x_lengths)-1))

    if is_sorted:
        # Do nothing if it's already sorted
        sorted_data = data
        sorted_labels = labels
    else:
        # Sorted indeces to get the data sorted by sequence length
        data_sorted_idx = list(np.argsort(x_lengths)[::-1])

        # Sort the x_lengths array by descending sequence length
        x_lengths = [x_lengths[idx] for idx in data_sorted_idx]

        # Sort the data by descending sequence length
        sorted_data = data[data_sorted_idx, :, :]

        if labels is not None:
            # Sort the labels by descending sequence length
            sorted_labels = labels[data_sorted_idx, :]

    if labels is None:
        return sorted_data, x_lengths
    else:
        return sorted_data, sorted_labels,  x_lengths


def in_ipynb():
    '''Detect if code is running in a IPython notebook, such as in Jupyter Lab.'''
    try:
        return str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
    except:
        # Not on IPython if get_ipython fails
        return False


def iterations_loop(x, see_progress=True):
    '''Determine if a progress bar is shown or not.'''
    if see_progress:
        # Define the method to use as a progress bar, depending on whether code
        # is running on a notebook or terminal
        if in_ipynb():
            return tqdm_notebook(x)
        else:
            return tqdm(x)
    else:
        # Don't show any progress bar if see_progress is False
        return x


def pad_list(x_list, length, padding_value=999999):
    '''Pad a list with a specific padding value until the desired length is
    met.

    Parameters
    ----------
    x_list : list
        List which will be padded.
    length : int
        Desired length for the final padded list.
    padding_value :
        Value to use in the padding, to fill the list.

    Returns
    -------
    x_list : list
        Resulting padded list'''
    return x_list + [padding_value] * (length - len(x_list))


def set_bar_color(values, ids, seq_len, threshold=0,
                  neg_color='rgba(30,136,229,1)', pos_color='rgba(255,13,87,1)'):
    '''Determine each bar's color in a bar chart, according to the values being
    plotted and the predefined threshold.

    Parameters
    ----------
    values : numpy.Array
        Array containing the values to be plotted.
    ids : int or list of ints
        ID or list of ID's that select which time series / sequences to use in
        the color selection.
    seq_len : int or list of ints
        Single or multiple sequence lengths, which represent the true, unpadded
        size of the input sequences.
    threshold : int or float, default 0
        Value to use as a threshold in the plot's color selection. In other
        words, values that exceed this threshold will have one color while the
        remaining have a different one, as specified in the parameters.
    pos_color : string
        Color to use in the bars corresponding to threshold exceeding values.
    neg_color : string
        Color to use in the bars corresponding to values bellow the threshold.

    Returns
    -------
    colors : list of strings
        Resulting bar colors list.'''
    if type(ids) is list:
        # Create a list of lists, with the colors for each sequences' instances
        return [[pos_color if val > 0 else neg_color for val in values[id, :seq_len]]
                for id in ids]
    else:
        # Create a single list, with the colors for the sequence's instances
        return [pos_color if val > 0 else neg_color for val in values[ids, :seq_len]]


def find_subject_idx(data, subject_id, subject_id_col=0):
    '''Find the index that corresponds to a given subject in a data tensor.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the data on which the subject's index will be
        searched for.
    subject_id : int or string
        Unique identifier of the subject whose index on the data tensor one
        wants to find out.
    subject_id_col : int, default 0
        The number of the column in the data tensor that stores the subject
        identifiers.

    Returns
    -------
    idx : int
        Index where the specified subject appears in the data tensor.'''
    return (data[:, 0, subject_id_col] == subject_id).nonzero().item()


def change_grad(grad, data, min=0, max=1):
    '''Restrict the gradients to only have valid values.

    Parameters
    ----------
    grad : torch.Tensor
        PyTorch tensor containing the gradients of the data being optimized.
    data : torch.Tensor
        PyTorch tensor containing the data being optimized.
    min : int, default 0
        Minimum valid data value.
    max : int, default 0
        Maximum valid data value.

    Returns
    -------
    grad : torch.Tensor
        PyTorch tensor containing the corrected gradients of the data being
        optimized.
    '''
    for i in range(data.shape[0]):
        if (data[i] == min and grad[i] < 0) or (data[i] == max and grad[i] > 0):
            # Stop the gradient from excedding the limit
            grad[i] = 0
        elif data[i] == min and grad[i] > 0.001:
            # Make the gradient have a integer value
            grad[i] = 1
        elif data[i] == max and grad[i] < -0.001:
            # Make the gradient have a integer value
            grad[i] = -1
        else:
            # Avoid any insignificant gradient
            grad[i] = 0

    return grad


def ts_tensor_to_np_matrix(data, feat_num=None, padding_value=999999):
    '''Convert a 3D PyTorch tensor, such as one representing multiple time series
    data, into a 2D NumPy matrix. Can be useful for applying the SHAP Kernel
    Explainer.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the three dimensional data being converted.
    feat_num : list of int, default None
        List of the column numbers that represent the features. If not specified,
        all columns will be used.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    data_matrix : numpy.ndarray
        NumPy two dimensional matrix obtained from the data after conversion.
    '''
    # View as a single sequence, i.e. like a dataframe without grouping by id
    data_matrix = data.contiguous().view(-1, data.shape[2]).detach().numpy()
    # Remove rows that are filled with padding values
    if feat_num is not None:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix[:, feat_num]]]
    else:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix]]
    return data_matrix


def model_inference(model, seq_len_dict, dataloader=None, data=None, metrics=['loss', 'accuracy', 'AUC'],
                    padding_value=999999, output_rounded=False, experiment=None, set_name='test',
                    seq_final_outputs=False, cols_to_remove=[0, 1]):
    '''Do inference on specified data using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which does the inference on the data.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches during inference.
    data : tuple of torch.Tensor, default None
        If a data loader isn't specified, the user can input directly a
        tuple of PyTorch tensor on which inference will be done. The first
        tensor must correspond to the features tensor whe second one
        should be the labels tensor.
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (loss), accuracy, AUC
        (Receiver Operating Curve Area Under the Curve), precision, recall
        and F1.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    output_rounded : bool, default False
        If True, the output is rounded, to represent the class assigned by
        the model, instead of just probabilities (>= 0.5 rounded to 1,
        otherwise it's 0)
    experiment : comet_ml.Experiment, default None
        Represents a connection to a Comet.ml experiment to which the
        metrics performance is uploaded, if specified.
    set_name : str
        Defines what name to give to the set when uploading the metrics
        values to the specified Comet.ml experiment.
    seq_final_outputs : bool, default False
        If set to true, the function only returns the ouputs given at each
        sequence's end.
    cols_to_remove : list of ints, default [0, 1]
        List of indeces of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as subject_id
        and ts (timestamp).

    Returns
    -------
    output : torch.Tensor
        Contains the output scores (or classes, if output_rounded is set to
        True) for all of the input data.
    metrics_vals : dict of floats
        Dictionary containing the calculated performance on each of the
        specified metrics.
    '''
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()

    # Create an empty dictionary with all the possible metrics
    metrics_vals = {'loss': None,
                    'accuracy': None,
                    'AUC': None,
                    'precision': None,
                    'recall': None,
                    'F1': None}

    # Initialize the metrics
    if 'loss' in metrics:
        loss = 0
    if 'accuracy' in metrics:
        acc = 0
    if 'AUC' in metrics:
        auc = 0
    if 'precision' in metrics:
        prec = 0
    if 'recall' in metrics:
        rcl = 0
    if 'F1' in metrics:
        f1_score = 0

    # Check if the user wants to do inference directly on a PyTorch tensor
    if dataloader is None and data is not None:
        features, labels = data[0].float(), data[1].float()             # Make the data have type float instead of double, as it would cause problems
        features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

        # Remove unwanted columns from the data
        features_idx = list(range(features.shape[2]))
        [features_idx.remove(column) for column in cols_to_remove]
        features = features[:, :, features_idx]
        scores = model.forward(features, x_lengths)                     # Feedforward the data through the model

        # Adjust the labels so that it gets the exact same shape as the predictions
        # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
        labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

        mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
        unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
        unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
        pred = torch.round(unpadded_scores)                             # Get the predictions

        if output_rounded:
            # Get the predicted classes
            output = pred.int()
        else:
            # Get the model scores (class probabilities)
            output = unpadded_scores

        if seq_final_outputs:
            # Only get the outputs retrieved at the sequences' end
            # Cumulative sequence lengths
            final_seq_idx = np.cumsum(x_lengths) - 1

            # Get the outputs of the last instances of each sequence
            output = output[final_seq_idx]

        if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
            # Calculate the number of true positives, false negatives, true negatives and false positives
            true_pos = int(sum(torch.masked_select(pred, unpadded_labels.byte())))
            false_neg = int(sum(torch.masked_select(pred == 0, unpadded_labels.byte())))
            true_neg = int(sum(torch.masked_select(pred == 0, (unpadded_labels == 0).byte())))
            false_pos = int(sum(torch.masked_select(pred, (unpadded_labels == 0).byte())))

        if 'loss' in metrics:
            metrics_vals['loss'] = model.loss(scores, labels, x_lengths).item() # Add the loss of the current batch
        if 'accuracy' in metrics:
            correct_pred = pred == unpadded_labels                          # Get the correct predictions
            metrics_vals['accuracy'] = torch.mean(correct_pred.type(torch.FloatTensor)).item() # Add the accuracy of the current batch, ignoring all padding values
        if 'AUC' in metrics:
            metrics_vals['AUC'] = roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the ROC AUC of the current batch
        if 'precision' in metrics:
            curr_prec = true_pos / (true_pos + false_pos)
            metrics_vals['precision'] = curr_prec                           # Add the precision of the current batch
        if 'recall' in metrics:
            curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['recall'] = curr_rcl                               # Add the recall of the current batch
        if 'F1' in metrics:
            # Check if precision has not yet been calculated
            if 'curr_prec' not in locals():
                curr_prec = true_pos / (true_pos + false_pos)
            # Check if recall has not yet been calculated
            if 'curr_rcl' not in locals():
                curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['F1'] = 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl) # Add the F1 score of the current batch

        return output, metrics_vals

    # Initialize the output
    output = torch.tensor([]).int()

    # Evaluate the model on the set
    for features, labels in dataloader:
        # Turn off gradients, saves memory and computations
        with torch.no_grad():
            features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
            features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

            # Remove unwanted columns from the data
            features_idx = list(range(features.shape[2]))
            [features_idx.remove(column) for column in cols_to_remove]
            features = features[:, :, features_idx]
            scores = model.forward(features, x_lengths)                     # Feedforward the data through the model

            # Adjust the labels so that it gets the exact same shape as the predictions
            # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
            labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
            labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

            mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
            unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
            unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
            pred = torch.round(unpadded_scores)                             # Get the predictions

            if output_rounded:
                # Get the predicted classes
                output = torch.cat([output, pred.int()])
            else:
                # Get the model scores (class probabilities)
                output = torch.cat([output.float(), unpadded_scores])

            if seq_final_outputs:
                # Indeces at the end of each sequence
                final_seq_idx = [n_subject*features.shape[1]+x_lengths[n_subject]-1 for n_subject in range(features.shape[0])]

                # Get the outputs of the last instances of each sequence
                output = output[final_seq_idx]

            if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
                # Calculate the number of true positives, false negatives, true negatives and false positives
                true_pos = int(sum(torch.masked_select(pred, unpadded_labels.byte())))
                false_neg = int(sum(torch.masked_select(pred == 0, unpadded_labels.byte())))
                true_neg = int(sum(torch.masked_select(pred == 0, (unpadded_labels == 0).byte())))
                false_pos = int(sum(torch.masked_select(pred, (unpadded_labels == 0).byte())))

            if 'loss' in metrics:
                loss += model.loss(scores, labels, x_lengths)               # Add the loss of the current batch
            if 'accuracy' in metrics:
                correct_pred = pred == unpadded_labels                      # Get the correct predictions
                acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the accuracy of the current batch, ignoring all padding values
            if 'AUC' in metrics:
                auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the ROC AUC of the current batch
            if 'precision' in metrics:
                curr_prec = true_pos / (true_pos + false_pos)
                prec += curr_prec                                           # Add the precision of the current batch
            if 'recall' in metrics:
                curr_rcl = true_pos / (true_pos + false_neg)
                rcl += curr_rcl                                             # Add the recall of the current batch
            if 'F1' in metrics:
                # Check if precision has not yet been calculated
                if 'curr_prec' not in locals():
                    curr_prec = true_pos / (true_pos + false_pos)
                # Check if recall has not yet been calculated
                if 'curr_rcl' not in locals():
                    curr_rcl = true_pos / (true_pos + false_neg)
                f1_score += 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl) # Add the F1 score of the current batch

    # Calculate the average of the metrics over the batches
    if 'loss' in metrics:
        metrics_vals['loss'] = loss / len(dataloader)
        metrics_vals['loss'] = metrics_vals['loss'].item()                  # Get just the value, not a tensor
    if 'accuracy' in metrics:
        metrics_vals['accuracy'] = acc / len(dataloader)
        metrics_vals['accuracy'] = metrics_vals['accuracy'].item()          # Get just the value, not a tensor
    if 'AUC' in metrics:
        metrics_vals['AUC'] = auc / len(dataloader)
    if 'precision' in metrics:
        metrics_vals['precision'] = prec / len(dataloader)
    if 'recall' in metrics:
        metrics_vals['recall'] = rcl / len(dataloader)
    if 'F1' in metrics:
        metrics_vals['F1'] = f1_score / len(dataloader)

    if experiment is not None:
        # Log metrics to Comet.ml
        if 'loss' in metrics:
            experiment.log_metric(f'{set_name}_loss', metrics_vals['loss'])
        if 'accuracy' in metrics:
            experiment.log_metric(f'{set_name}_acc', metrics_vals['accuracy'])
        if 'AUC' in metrics:
            experiment.log_metric(f'{set_name}_auc', metrics_vals['AUC'])
        if 'precision' in metrics:
            experiment.log_metric(f'{set_name}_prec', metrics_vals['precision'])
        if 'recall' in metrics:
            experiment.log_metric(f'{set_name}_rcl', metrics_vals['recall'])
        if 'F1' in metrics:
            experiment.log_metric(f'{set_name}_f1_score', metrics_vals['F1'])

    return output, metrics_vals


def train(model, train_dataloader, val_dataloader, seq_len_dict,
          batch_size=32, n_epochs=50, lr=0.001, model_path='models/',
          padding_value=999999):
    '''Trains a given model on the provided data.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which is trained on the data to perform a
        classification task.
    train_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches during training.
    val_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches when evaluating
        the model's performance on a validation set during training.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    n_epochs : int, default 50
        Number of epochs, i.e. the number of times the training loop
        iterates through all of the training data.
    lr : float, default 0.001
        Learning rate used in the optimization algorithm.
    model_path : string, default 'models/'
        Path where the model will be saved. By default, it saves in
        the directory named "models".
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    model : nn.Module
        The same input model but with optimized weight values.
    '''
    optimizer = optim.Adam(model.parameters(), lr=lr)                       # Adam optimization algorithm
    step = 0                                                                # Number of iteration steps done so far
    print_every = 10                                                        # Steps interval where the metrics are printed
    train_on_gpu = torch.cuda.is_available()                                # Check if GPU is available
    val_loss_min = np.inf                                                   # Start with an infinitely big minimum validation loss

    for epoch in range(1, n_epochs+1):
        # Initialize the training metrics
        train_loss = 0
        train_acc = 0
        train_auc = 0

        # try:
        # Loop through the training data
        for features, labels in train_dataloader:
            model.train()                                                   # Activate dropout to train the model
            optimizer.zero_grad()                                           # Clear the gradients of all optimized variables

            if train_on_gpu:
                features, labels = features.cuda(), labels.cuda()           # Move data to GPU

            features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
            features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length
            scores = model.forward(features[:, :, 2:], x_lengths)           # Feedforward the data through the model
                                                                            # (the 2 is there to avoid using the identifier features in the predictions)

            # Adjust the labels so that it gets the exact same shape as the predictions
            # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
            labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
            labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

            loss = model.loss(scores, labels, x_lengths)                    # Calculate the cross entropy loss
            loss.backward()                                                 # Backpropagate the loss
            optimizer.step()                                                # Update the model's weights
            train_loss += loss                                              # Add the training loss of the current batch
            mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
            unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
            unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
            pred = torch.round(unpadded_scores)                             # Get the predictions
            correct_pred = pred == unpadded_labels                          # Get the correct predictions
            train_acc += torch.mean(correct_pred.type(torch.FloatTensor))   # Add the training accuracy of the current batch, ignoring all padding values
            train_auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the training ROC AUC of the current batch
            step += 1                                                       # Count one more iteration step
            model.eval()                                                    # Deactivate dropout to test the model

            # Initialize the validation metrics
            val_loss = 0
            val_acc = 0
            val_auc = 0

            # Loop through the validation data
            for features, labels in val_dataloader:
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
                    features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length
                    scores = model.forward(features[:, :, 2:], x_lengths)           # Feedforward the data through the model
                                                                                    # (the 2 is there to avoid using the identifier features in the predictions)

                    # Adjust the labels so that it gets the exact same shape as the predictions
                    # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
                    labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
                    labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

                    val_loss += model.loss(scores, labels, x_lengths)               # Calculate and add the validation loss of the current batch
                    mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
                    unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
                    unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
                    pred = torch.round(unpadded_scores)                             # Get the predictions
                    correct_pred = pred == unpadded_labels                          # Get the correct predictions
                    val_acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the validation accuracy of the current batch, ignoring all padding values
                    val_auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the validation ROC AUC of the current batch

            # Calculate the average of the metrics over the batches
            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            val_auc = val_auc / len(val_dataloader)

            # Display validation loss
            if step%print_every == 0:
                print(f'Epoch {epoch} step {step}: Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')

            # Check if the performance obtained in the validation set is the best so far (lowest loss value)
            if val_loss < val_loss_min:
                print(f'New minimum validation loss: {val_loss_min} -> {val_loss}.')

                # Update the minimum validation loss
                val_loss_min = val_loss

                # Get the current day and time to attach to the saved model's name
                current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

                # Filename and path where the model will be saved
                model_filename = f'{model_path}checkpoint_{current_datetime}.pth'

                print(f'Saving model in {model_filename}')

                # Save the best performing model so far, a long with additional information to implement it
                checkpoint = {'n_inputs': model.n_inputs,
                              'n_hidden': model.n_hidden,
                              'n_outputs': model.n_outputs,
                              'n_layers': model.n_layers,
                              'p_dropout': model.p_dropout,
                              'state_dict': model.state_dict()}
                torch.save(checkpoint, model_filename)

        # Calculate the average of the metrics over the epoch
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        train_auc = train_auc / len(train_dataloader)

        # Print a report of the epoch
        print(f'Epoch {epoch}: Training loss: {train_loss}; Training Accuracy: {train_acc}; Training AUC: {train_auc}; \
                Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')
        print('----------------------')
        # except:
        #     warnings.warn(f'There was a problem doing training epoch {epoch}. Ending training.')

    # Load the model with the best validation performance
    model = load_checkpoint(model_filename)

    return model
