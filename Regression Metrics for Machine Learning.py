# Import libraries
import numpy as np
from sklearn import metrics


# In[2]:


# RMSE
def rmse(y_true, y_pred):
    squared_diff = (y_true - y_pred) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse_value = np.sqrt(mean_squared_diff)
    return rmse_value


# In[3]:


# Example true and predicted values
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[4]:


rmse_value = rmse(y_true, y_pred)
print("RMSE:", rmse_value)
print(f"RMSE: {rmse_value * 100:.4f}", '%')


# In[5]:


#---------------------------------------------------------------------------------------------------
# RRMSE
def rrmse(y_true, y_pred):
    # Calculate the squared errors between the predicted and true values
    squared_errors = (y_true - y_pred) ** 2
    
    # Calculate the mean of the squared errors
    mean_squared_error = np.mean(squared_errors)
    
    # Take the square root of the mean squared error
    root_mean_squared_error = np.sqrt(mean_squared_error)
    
    # Calculate the relative error by dividing the root mean squared error by the mean of the true values
    relative_error = root_mean_squared_error / np.mean(y_true)
    
    # Return the RRMSE value
    return relative_error


# In[6]:


# Example true and predicted values
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[7]:


# Calculate the RRMSE
rrmse = rrmse(y_true, y_pred)
print(f"RRMSE: {rrmse:.4f}")
print(f"RRMSE: {rrmse * 100:.4f}", '%')


# In[8]:


#---------------------------------------------------------------------------------------------------
# RSE
def root_squared_error(y_true, y_pred):
    """
    Calculate the Root Squared Error between two arrays (y_true and y_pred).
    
    Args:
        y_true (numpy.ndarray): Actual values.
        y_pred (numpy.ndarray): Predicted values.
        
    Returns:
        float: The Root Squared Error.
    """
    error = y_true - y_pred
    squared_error = np.square(error)
    mean_squared_error = np.mean(squared_error)
    root_squared_error = np.sqrt(mean_squared_error)
    
    return root_squared_error


# In[9]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[10]:


# Calculate RSE
rse = root_squared_error(y_true, y_pred)
print(f"RSE: {rse}")
print(f"RSE: {rse * 100:.4f}", '%')


# In[11]:


#---------------------------------------------------------------------------------------------------
# NSE
def nash_sutcliffe_efficiency(y_true, y_pred):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) between two arrays (y_true and y_pred).
    
    Args:
        y_true (numpy.ndarray): Actual values.
        y_pred (numpy.ndarray): Predicted values.
        
    Returns:
        float: The Nash-Sutcliffe Efficiency.
    """
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    nse = 1 - (numerator / denominator)
    
    return nse


# In[12]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[13]:


# Calculate NSE
nse = nash_sutcliffe_efficiency(y_true, y_pred)
print(f"NSE: {nse}")
print(f"NSE: {nse * 100:.4f}", '%')


# In[14]:


#---------------------------------------------------------------------------------------------------
# MAE
def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between two arrays (y_true and y_pred).
    
    Args:
        y_true (numpy.ndarray): Actual values.
        y_pred (numpy.ndarray): Predicted values.
        
    Returns:
        float: The Mean Absolute Error.
    """
    absolute_error = np.abs(y_true - y_pred)
    mae = np.mean(absolute_error)
    
    return mae


# In[15]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[16]:


# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae}")
print(f"MAE: {mae * 100:.4f}", '%')


# In[17]:


#---------------------------------------------------------------------------------------------------
# R
def pearson_correlation_coefficient(y_true, y_pred):
    """
    Calculate the Pearson Correlation Coefficient (R) between two arrays (y_true and y_pred).
    
    Args:
        y_true (numpy.ndarray): Actual values.
        y_pred (numpy.ndarray): Predicted values.
        
    Returns:
        float: The Pearson Correlation Coefficient.
    """
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r = correlation_matrix[0, 1]
    
    return r


# In[18]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[19]:


# Calculate Pearson Correlation Coefficient
r = pearson_correlation_coefficient(y_true, y_pred)
print(f"Pearson Correlation Coefficient (R): {r}")


# In[20]:


#---------------------------------------------------------------------------------------------------
# R2
def r_squared(y_true, y_pred):
    """
    Calculate the R squared value between two arrays (y_true and y_pred).
    
    Args:
        y_true (numpy.ndarray): Actual values.
        y_pred (numpy.ndarray): Predicted values.
        
    Returns:
        float: The R squared value.
    """
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    return r_squared


# In[21]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[22]:


# Calculate R squared
r2 = r_squared(y_true, y_pred)
print(f"R squared value: {r2}")


# In[23]:


#---------------------------------------------------------------------------------------------------
# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[24]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[25]:


# Calculate MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape}")


# In[26]:


#---------------------------------------------------------------------------------------------------
# ρ (RRMSE / (1 + R))
def relative_rmse(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse / (np.max(y_true) - np.min(y_true))

def pearson_correlation_coefficient(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r = correlation_matrix[0, 1]
    return r


# In[27]:


# Example data
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([1.8, 3.9, 6.2, 7.8, 9.7])


# In[28]:


# Calculate Relative RMSE and Pearson Correlation Coefficient
rrmse = relative_rmse(y_true, y_pred)
r = pearson_correlation_coefficient(y_true, y_pred)


# In[29]:


# Calculate RRMSE / (1 + R)
result = rrmse / (1 + r)
print(f"ρ (rho): {result}")

