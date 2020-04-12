import pandas as pd 
import numpy as np 

restaurants = pd.read_excel('FinalDataSet.xlsx')
restaurants = restaurants.to_numpy()
restaurants[:,-1] = restaurants[:, -1] * 2
np.save('FinalDataSet.npy', restaurants)