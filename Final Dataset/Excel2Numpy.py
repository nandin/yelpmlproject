import pandas as pd 
import numpy as np 

restaurants = pd.read_excel('RestaurantDataV8.xlsx')
restaurants = restaurants.to_numpy()
restaurants[:,-1] = restaurants[:, -1] * 2
np.save('FullData.npy', restaurants)