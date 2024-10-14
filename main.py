import numpy as np
import scipy as sp
import pandas as pd

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)

def Datamatrix_Y(str):
    """Loads the datamatrix and y vector from csv file"""
    csv_data = pd.read_csv('data.csv')
    X_matrix = csv_data.iloc[:, :-1].values
    y_vector=csv_data.iloc[:,-1].values
    X_matrix=np.array(X_matrix)
    y_vector=np.array(y_vector)
    return X_matrix,y_vector


X_mat,y_vec=Datamatrix_Y('data.csv')


print(X_mat)
print(X_mat.shape)

#Calculate sparsity
sparsity=len(X_mat[X_mat==0])/len(X_mat)
print(f"{round(sparsity,2)} % of the entries are zero's")

