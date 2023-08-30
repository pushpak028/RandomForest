import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('\file_path or file_name')

x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

rf = RandomForestRegressor(n_estimator = 10 , random_state =0)
rf.fit(x,y)

#lets check for an example of 6.5 level from the dataset

rf.predict([[6.5]])

#o/p would be 167000

#lets plot 

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,rf.predict(x_grid))
