import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # or 'Agg' for non-interactive backend
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def Datamatrix_Y(str):
    """Loads the datamatrix and y vector from csv file"""
    csv_data = pd.read_csv('data.csv')
    print(csv_data)
    X_matrix = csv_data.iloc[:, :-1].values
    y_vector=csv_data.iloc[:,-1].values

    X_matrix=np.array(X_matrix)
    y_vector=np.array(y_vector)
    print(y_vector)
    return X_matrix,y_vector


X_mat,y_vec=Datamatrix_Y('data.csv')
csv_data = pd.read_csv('data.csv')
csv_data=np.array(csv_data)
print("-1", csv_data[csv_data==-1])
plt.plot(X_mat)
plt.show()


print(X_mat)
print(X_mat.shape)

#Calculate sparsity
sparsity=len(X_mat[X_mat==0])/len(X_mat)
print(f"{round(sparsity,2)} % of the entries are zero's")

