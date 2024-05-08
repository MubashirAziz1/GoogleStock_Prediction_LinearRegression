import pandas as pd
import datetime
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv('D:\AI\ML\Google_Stock\GOOG.csv')
df= df[['date', 'adjOpen', 'adjClose', 'adjHigh', 'adjLow', 'adjVolume']]

# date Series or df['date'] is a combination of date and time in string 
# datatype. Only the date column o_date is taken out by using split method and 
# the other columns are removed.

df[['o_date', 'time']] = df['date'].str.split(' ', expand = True)
df.drop('date', axis = 1,inplace = True)
df.drop('time', axis = 1,inplace = True)

# New Features are added to improve the performace of ML model.
df['HL_PCT'] = ((df['adjHigh'] - df['adjClose']) / df['adjClose'])*100
df['PCT_Change'] = ((df['adjClose'] - df['adjOpen']) / df['adjOpen'])*100
df= df[['o_date', 'adjClose','HL_PCT', 'PCT_Change', 'adjVolume']]

# Select the Column which we want to predict and fill the Null tabs by outliers.
forecast_col = 'adjClose'
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

# Create the Feature and the Label for the model.
x_columns = ['adjClose', 'HL_PCT', 'PCT_Change', 'adjVolume']
x = np.array(df[x_columns])
x = preprocessing.scale(x)
# Train the only available Features
x = x[ : -forecast_out]
# At last store the features of forecast_out days in a seperate variable to predict at the end.
x_lately = x[-forecast_out : ]
df.dropna(inplace = True)
y = np.array(df['label'])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
## Picke it to store the trained data so everytime we dont train the data again
## but reuse the trained data 

#reg = linear_model.LinearRegression()
#reg.fit(x_train, y_train)
#with open('GS_LR.pickle','wb') as f:
 #   pickle.dump(reg, f)
pickle_in = open('GS_LR.pickle','rb')

reg = pickle.load(pickle_in)
forecast_set = reg.predict(x_lately)
df['Forecast'] = np.nan

df['o_date'] = pd.to_datetime(df['o_date'])
df.set_index(df['o_date'], inplace=True)
last_date = df.index[-1]
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    
df = df[['adjClose', 'HL_PCT', 'PCT_Change', 'adjVolume','label', 'Forecast']]
df['adjClose'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



