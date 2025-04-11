import pandas as pd
import numpy as np

def load_data():
    """
    Load training, development, and test datasets from the 'splits' directory.
    Also, load the test labels for evaluation.

     Returns:
    - train_data: DataFrame containing the training data.
    - dev_data: DataFrame containing the development (or validation) data.
    - test_data: DataFrame containing the test data.
    - test_labels: DataFrame containing the "year" label corresponding to the test data.
    - hidden_test_data: DataFrane containing the test data for the kaggle submission.
    """
    train_data = pd.read_csv('../splits/train_data.csv', header=None)
    dev_data = pd.read_csv('../splits/dev_data.csv', header=None)
    hidden_test_data = pd.read_csv('../splits/hidden_test_data.csv', header=None)
    test_data = pd.read_csv('../splits/test_data.csv', header=None)
    test_labels = pd.read_csv('../splits/test_labels.csv', header=None).values.ravel()  # load labels and flatten

    return train_data, dev_data, test_data, hidden_test_data, test_labels

def prepare_data(data):
    """
    Extract features and targets from the data and add a bias term to the features.
    
    Args:
    - data: DataFrame containing the dataset.
    
    Returns:
    - X: Array containing the feature vectors with an added bias term.
    - y: Array containing the target values (None if data doesn't have targets).
    """
    if data.shape[1] > 90:  # Check if the data contains target column
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
    else:
        X = data
        y = None
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding bias term
    return X, y


def transform_features(X,mu,s):
    '''
    For Q3
    Args:
    - X: Array containing the feature vectors.
    - mu,s : Hyperparameters for the radial basis function
    
    Returns:
    - X_tf : Array containing the feature vectors with the transformed features concatenated
    '''
    #TODO: Implement the radial basis function transformation
    X_rbf = None #Add code to apply RBF transformation on X
    X_tf = None # Add code to concatenate X and X_rbf along appropriate axis
    return X_tf

def train_model(X_train, y_train):
    """
    Train a linear regression model using the closed-form solution.
    
    Args:
    - X_train: Array containing the training feature vectors.
    - y_train: Array containing the training target values.
    
    Returns:
    - theta: Array containing the learned weights (including bias).
    """
    # TODO: Fill in the closed-form solution for linear regression, a single line of code!
    # Use pinverse instead of inverse (especially important when using basis functions) 
    theta = None
    return theta

def predict(X, theta):
    """
    Predict the target values using the learned weights.
    
    Args:
    - X: Array containing the feature vectors to predict on.
    - theta: Array containing the learned weights (including bias).
    
    Returns:
    - y_pred: Array containing the predicted target values.
    """
    # TODO: Use theta to make predictions on X, a single line of code!
    y_pred = None
    return y_pred

def calculate_errors(y_pred, y_true):
    """
    Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
    
    Args:
    - y_pred: Array containing the predicted target values.
    - y_true: Array containing the actual target values.
    
    Returns:
    - mse: Mean Squared Error.
    - rmse: Root Mean Squared Error.
    """
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    return mse, rmse


if __name__ == "__main__":
    # Load datasets
    train_data, dev_data, test_data, hidden_test_data, test_labels = load_data()

    # Prepare data by extracting features and target variables
    X_train, y_train = prepare_data(train_data)
    X_dev, y_dev = prepare_data(dev_data)
    X_test, _ = prepare_data(test_data)
    X_hidden_test, _ = prepare_data(hidden_test_data)

    '''
    Uncomment the lines below for Q3 
    For Q3, we will use basis functions to try and improve over simple linear regression
    We will use radial basis functions concatenated with original features (refer pdf for more details)
    '''
    # mu = None #TODO : Calculate the mean of training features, use np.mean() along appropriate axis
    # s = 1 # Hyperparameter - you can tune this to improve fit
    
    # X_train = transform_features(X_train, mu, s)
    # X_dev = transform_features(X_dev, mu, s)
    # X_test = transform_features(X_test, mu, s)
    # X_hidden_test = transform_features(X_hidden_test, mu, s)
    
    # Train the model to get the parameters (theta)
    theta = train_model(X_train, y_train)

    # Validate the model using the dev set
    y_pred_dev = predict(X_dev, theta)
    mse, rmse = calculate_errors(y_pred_dev, y_dev)
    print(f"Dev Mean Squared Error: {mse}")
    print(f"Dev Root Mean Squared Error: {rmse}")

    # Calculate RMSE for test set using test_labels
    y_pred_test_for_rmse = predict(X_test, theta)
    _, test_rmse = calculate_errors(y_pred_test_for_rmse, test_labels)
    print(f"Test Root Mean Squared Error: {test_rmse}")

    '''
    Uncomment the section below for Question 3 
    This saves predictions on the hidden test set to a csv file
    ''' 
    # # Predict labels for the hidden test set
    # y_pred_hidden_test = predict(X_hidden_test, theta)

    # # Save predictions
    # df = pd.DataFrame(y_pred_hidden_test, columns=['Year'])
    # df.index.name = 'ID'
    # df.to_csv(
    #     'roll_number.csv', index=True)
