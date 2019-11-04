#for ML algo
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#for data visualization
import matplotlib.pyplot as plt 
from matplotlib import style
import seaborn as sns

#for data reading
import datetime as dt 
import pandas_datareader.data as web

#micro libraries
import datetime
import math

'''
start = dt.datetime(2016, 1, 1)
end = dt.datetime(2019, 10, 10)

# imports data from web
df = web.DataReader('TSLA', 'yahoo', start, end)

#creates a csv file
df.to_csv('tsla1.csv')
'''

df = pd.read_csv("tsla1.csv", parse_dates = True, index_col = 0)
df.fillna(-99999, inplace = True)#filling NaN
#data visualization
print(df.head())
print(df.dtypes)
print(df.shape)

'''
style.use("ggplot")
df[['Adj Close', 'High']].plot()
plt.legend()
plt.show()
'''

'''
sns.pairplot(df)
plt.suptitle('Pair Plot of Data for 2015-2019', 
             size = 20);
plt.show()
'''

# creating forecasting column
'''
forcast_out = int(math.ceil(0.01 * len(df)))
print(forcast_out) # forecasting .01 fraction of the total days of dataset
'''

forcast_out = 7
df['Prediction'] = df[['Adj Close']].shift(-forcast_out)
#print(df.head())
print(df.tail(forcast_out+1)) # prediction columns will be filled by NaN

#creating independent and dependent dataset
x = np.array(df.drop(['Prediction'], 1))
y = np.array(df['Prediction'])

#processing
x = preprocessing.scale(x)
x_val = x[-forcast_out:] #final values 
x = x[:-forcast_out] # removing last forecasting days
y = y[:-forcast_out] # removing last forecasting days


# data split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) #len(x) must be == len(y)
#print(len(x), len(y))

#classifier
clf = LinearRegression(n_jobs = -1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

'''
After calculation , we can drop the 7 day ahead prediction
'''
df.drop(['Prediction'],axis =1, inplace = True)
#print(df.tail())

#results
print("accuracy of the model in r^2 :", accuracy)
y_val = clf.predict(x_val)
print("next ", forcast_out, "days forecast:\n", y_val)

#plotting new data

df["Forcast"] = np.nan 

last_date = df.iloc[-forcast_out].name
#print(last_date)

last_unix = last_date.timestamp()
#print(last_unix)
one_day = 86400
next_unix = last_unix + one_day

for i in y_val:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

#all the forcast will be nan except the last forcast value()
#all the columns will be nan for last 7 days except the forcast column
print(df.tail(10))


df["Adj Close"].plot()
df["Forcast"].plot()
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

