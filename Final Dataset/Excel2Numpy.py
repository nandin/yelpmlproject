import pandas as pd 
import numpy as np 

restaurants = pd.read_excel('FinalDataSet.xlsx')
restaurants = restaurants.to_numpy()
np.save('FinalDataSet.npy', restaurants)