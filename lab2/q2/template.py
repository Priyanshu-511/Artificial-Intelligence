import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

class LinearRegressionBatchGD:
  def __init__(self, learning_rate=0.01, max_epochs=200, batch_size=None):
    '''
    Initializing the parameters of the model
    
    Args:
      learning_rate : learning rate for batch gradient descent
      max_epochs : maximum number of epochs that the batch gradient descent algorithm will run for
      batch-size : size of the batches used for batch gradient descent.

    Returns: 
      None 
    '''
    self.learning_rate = learning_rate
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.weights = None
    
  def fit(self, X, y, X_dev, y_dev):
    '''
    This function is used to train the model using batch gradient descent.

    Args:
      X : 2D numpy array of training set data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
      X_dev : 2D numpy array of development set data points. Dimensions (n x (d+1))
      y_dev : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
      
    Returns : 
      None
    '''
    if self.batch_size is None:
      self.batch_size = X.shape[0]

    # Initialize the weights
    self.weights = np.zeros((X.shape[1],1))
    
    prev_weights = self.weights
    
    self.error_list = []  #stores the loss for every epoch 
    for epoch in range(self.max_epochs):
      
      batches = create_batches(X, y, self.batch_size)
      
      for batch in batches:
        X_batch, y_batch = batch  #X_batch and y_batch are data points and target values for a given batch
        
      # Complete the inner "for" loop to calculate the gradient of loss w.r.t weights, i.e. dw and update the weights
      # You should use "compute_gradient()"  function to calculate gradient.
      
          
      # After the inner "for" loop ends, calculate loss on the validation/development set using "compute_rmse_loss()" function and add the loss of each epoch to the "error list"
      
      
      if np.linalg.norm(self.weights - prev_weights) < 1e-5:
        print(f" Stopping at epoch {epoch}.")
        break

    print("Training complete.")
    print("Mean validation RMSE loss : ", np.mean(self.error_list))
    print("Batch size: ", self.batch_size)
    print("learning rate: ", self.learning_rate)

    plot_loss(self.error_list,self.batch_size)
    
  def predict(self, X):
    '''
    This function is used to predict the target values for the given set of feature values

    Args:
      X: 2D numpy array of data points. Dimensions (n x (d+1)) 
    
    Returns:
      2D numpy array of predicted target values. Dimensions (n x 1)
    '''
    # Write your code here

  def compute_rmse_loss(self, X,y,weights):
    '''
    This function computes the Root Mean Square Error (RMSE) 

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)
      weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)
    
    Returns:
      loss : 2D numpy array of RMSE loss. Dimensions (1x1)
    '''
    # Write your code here

  def compute_gradient(self, X,y,weights):
    '''
    This function computes the gradient of mean squared-error loss w.r.t the weights

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)
      weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)
    
    Returns:
      dw : 2D numpy array of gradients w.r.t weights. Dimensions ((d+1) x 1)
    '''
    # Write your code here.
    # Note: Make sure you divide the gradient (dw) by the total number of training instances before returning to prevent "exploding gradients".

def plot_loss(error_list, batch_size):
  '''
  This function plots the loss for each epoch.

  Args:
    error_list : list of validation loss for each epoch
    batch_size : size of one batch
  Returns:
    None
  '''
  # Complete this function to plot the graph of losses stored in model's "error_list"
  # Save the plot in "figures" folder.
  

def standard_scaler(data):
  '''
  This function is used to Scale the values of the argument down to the range [0,1]

  Args:
    data : numpy array of values
  
  Returns:
    standardized_data : numpy array of original dimension with scaled down values.
    
  '''
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
    
  standardized_data = (data - mean) / std
  return standardized_data


def create_batches(X, y, batch_size):
  '''
  This function is used to create the batches of randomly selected data points.

  Args:
    X : 2D numpy array of data points. Dimensions (n x (d+1))
    y : 2D numpy array of target values. Dimensions (n x 1)

  Returns:
    batches : list of tuples with each tuple of size batch size.
  '''
  batches = []
  data = np.hstack((X, y))
  np.random.shuffle(data)
  num_batches = data.shape[0]//batch_size 
  i = 0
  for i in range(num_batches+1):
    if i<num_batches:
      batch = data[i * batch_size:(i + 1)*batch_size, :]
      X_batch = batch[:, :-1]
      Y_batch = batch[:, -1].reshape((-1, 1))
      batches.append((X_batch, Y_batch))
    if data.shape[0] % batch_size != 0 and i==num_batches:
      batch = data[i * batch_size:data.shape[0]]
      X_batch = batch[:, :-1]
      Y_batch = batch[:, -1].reshape((-1, 1))
      batches.append((X_batch, Y_batch))
  return batches


def load_train_dev_dataset():
  '''
  This function loads the dataset and Normalizes the feature and target values that lie in between 0 and 1.

  Args:
    None.

  Returns: 
    X_train : 2D numpy array of training set data points. Dimensions (n x (d+1))
    y_train : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
    X_dev : 2D numpy array of development set data points. Dimensions (n x (d+1))
    y_dev : 2D numpy array of target values in the training dataset. Dimensions (n x 1) 
    y_train_min : smallest value in the target labels array of training dataset.
    y_train_max : largest value in the target labels array of training dataset.
  '''
  train_set = pd.read_csv(f"./splits/train_data.csv",header = None)
  dev_set = pd.read_csv(f"./splits/dev_data.csv", header = None)

  # splitting train dataset
  X_train = train_set.iloc[:,1:].to_numpy()
  y_train = train_set.iloc[:,0].to_numpy().reshape(-1,1)
  y_train_mean, y_train_std = np.mean(y_train,axis=0), np.std(y_train, axis=0)
  
  #splitting dev dataset
  X_dev = dev_set.iloc[:,1:].to_numpy()
  y_dev = dev_set.iloc[:,0].to_numpy().reshape(-1,1)

  #Scaling the values to range [0,1]
  X_train, y_train, X_dev, y_dev = scaling(X_train, y_train, X_dev, y_dev)

  #adding a column to X_train and X_dev 
  X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
  X_dev = np.c_[np.ones((X_dev.shape[0], 1)), X_dev]
  
  
  return X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std

def scaling(X_train, y_train, X_dev, y_dev):
  '''
  This function is used for normalizing the datasets.

  Args:
    X_train : 2D numpy array of training set data points. Dimensions (n x (d+1))
    y_train : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
    X_dev : 2D numpy array of development set data points. Dimensions (n x (d+1))
    y_dev : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
  
  Returns:
    X_train, y_train, X_dev, y_dev

  '''
  X_train = standard_scaler(X_train)
  y_train = standard_scaler(y_train)
  X_dev = standard_scaler(X_dev)
  y_dev = standard_scaler(y_dev)
  return X_train, y_train, X_dev, y_dev 

def load_test_dataset():
  '''
  This function is used to load the Test dataset in the memory.

  Args: 
    None.
  
  Returns:
    X_train : 2D numpy array of test set data points. Dimensions (n x (d+1))
    y_train : 2D numpy array of target values in the test dataset. Dimensions (n x 1)
  '''
  X_test = pd.read_csv(f"./splits/test_data.csv",header = None).to_numpy()
  y_test = pd.read_csv(f"./splits/test_labels.csv",header = None).to_numpy().reshape(-1,1)
  X_test = standard_scaler(X_test)
  X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
  return X_test, y_test

def evaluate_model(weights, X, y, ymean, ystd):
  '''
  This function is used to calculate the RMSE loss on test dataset.

  Args:
    weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)
    X : 2D numpy array of test set data points. Dimensions (n x (d+1))
    y : 2D numpy array of target values in the test dataset. Dimensions (n x 1)
    y_min : minimum value of target labels in the train set.
    y_max : maximum value of target labels in the train set.
  '''
  y_pred_scaled = X @ weights
  y_pred_actual = y_pred_scaled * ystd + ymean
  difference = (y_pred_actual) - y
  rmse = np.sqrt(np.mean(difference**2))
  return rmse

def save_prediction(ymean, ystd, weights):
  '''
  Function to save the model predictions on hiddent_test_dataset

  Args:
    ymean : mean value of target labels of train dataset
    ystd : standard deviation of target labels of train dataset

  Returns:
    None
  '''
  X = pd.read_csv(f"./splits/hidden_test_data.csv",header = None).to_numpy()
  X = standard_scaler(X)
  X = np.c_[np.ones((X.shape[0], 1)), X]
  predictions = (X @ weights)
  y_pred_hidden_test = (predictions * ystd + ymean)
  pd.DataFrame(y_pred_hidden_test, columns=['Year']).to_csv('roll_number.csv', index=True, header=True,index_label = "ID")


if __name__ == '__main__':

    
    # Hyperparameters
    learning_rate = 0.0001
    batch_size = None
    max_epochs = 150

    # loading the dataset
    X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std = load_train_dev_dataset()
    X_test, y_test = load_test_dataset()

    
    # Create LinearRegressionBatchGD instance
    model = LinearRegressionBatchGD(learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size)

    # Train the model
    model.fit(X_train, y_train, X_dev, y_dev)

    #printing the RMSE for test dataset
    print(evaluate_model(model.weights,X_test, y_test,y_train_mean, y_train_std))

    #saving the predictions for hidden test dataset
    save_prediction(y_train_mean, y_train_std,model.weights)
