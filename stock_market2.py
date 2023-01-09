#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the library
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly as py 
import cufflinks as cff
from numpy import sqrt,log
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,Holt,ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow


# In[2]:


df=pd.read_csv("tcs_data2.csv")


# In[3]:


df


# In[4]:


#data structure
print(type(df))
df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df=df.rename(columns={"Deliverable Volume":"Deliverablevolume","%Deliverble":"Deliverable"})


# In[9]:


df.duplicated().sum()


# In[10]:


print(plt.style.available)
plt.style.use('seaborn-deep')


# In[11]:


df_open=pd.DataFrame()
df_open["TCS"]=pd.Series(df["Open"])
df_open["Date"]=pd.Series(df["Date"])


# In[81]:


plt.rcParams["figure.figsize"]=(15,10)
sns.pairplot(df)


# In[12]:


# Relaince Variables
plt.figure(figsize=(15,6))
sns.heatmap(df.corr(),annot=True)


# ## Opening price of the stock when the market opens

# In[13]:


plt.rcParams["figure.figsize"]=(22,12)
df_open.plot(x="Date")
plt.ylabel("Price",size=25)
plt.legend(loc="upper left")


# In[14]:


df_close=pd.DataFrame()
df_close["TCS"]=pd.Series(df["Close"])
df_close["Date"]=pd.Series(df["Date"])


# In[15]:


plt.style.use("tableau-colorblind10")


# ## Closing price of the stock when the market closed

# In[16]:


plt.rcParams["figure.figsize"]=(22,9)
df_close.plot(x="Date")
plt.ylabel("Price",size=25)
plt.legend(loc="upper left")


# In[17]:


df_volume=pd.DataFrame()
df_volume["TCS"]=pd.Series(df["Volume"])
df_volume["Date"]=pd.Series(df["Date"])


# ## Total amount of stock traded on that day

# In[18]:


# Volume 
plt.style.use("tableau-colorblind10")
df_volume.plot(x="Date")
plt.legend(loc="upper left")
plt.ylabel("Count",size=23)


# In[19]:


# To identify the Trend and Sesonality
df_seasonal=seasonal_decompose(df.Close,period=50)

df_seasonal.plot()
# ACF Plot - to identify the corrrlation 
plot_acf(df["Close"],lags=30)

plt.show()
plt.rcParams["figure.figsize"]=(12,4)


# ## Moving average

# In[20]:


# Moving Average 
T_MA=pd.DataFrame()
T_MA["50"]=df["Close"].rolling(50).mean()
T_MA["100"]=df["Close"].rolling(100).mean()
T_MA["Date"]=df["Date"]
T_MA["Close"]=df["Close"]
T_MA=T_MA.set_index("Date")
label=["MA50","MA100","TCS Close"]
T_MA.plot()
plt.ylabel("Average")
plt.legend(label,loc="upper left")


# In[21]:


py.offline.init_notebook_mode(connected=True)
cff.go_offline()


# In[22]:


# TCS Year wise Average price
df_Quarter=pd.read_csv("tcs_data2.csv",parse_dates=["Date"],index_col="Date")
df_Quarter=df_Quarter.Close.resample('Y').mean()
df_Quarter.iplot(kind="line",color="black")


# In[23]:


df_high=pd.DataFrame()
df_high["TCS"]=df.High
# TCS high prices

# Making Date as a Index 
df_high["Date"]=df.Date
df_high=df_high.set_index("Date")
df_high.iplot(kind="bar")


# In[24]:


df_low=pd.DataFrame()
df_low["TCS"]=df.Low
# TCS low prices

# Making Date as a Index 
df_low["Date"]=df.Date
df_low=df_low.set_index("Date")
df_low.iplot(kind="bar")


# In[25]:


df_volatility=pd.DataFrame()
df_volatility["TCS"]=(df["Close"]/df["Close"].shift(1))-1
df_volatility["TCS"].hist(bins=200,alpha=0.7,label="TCS")
plt.legend()


# In[26]:


df["High"].max(),df["Low"].max()


# In[27]:


plt.style.use("classic")
plt.rcParams["figure.figsize"]=(18,4)
H_max_x = [4043.0,3980]
L_max_y = ["Highest Price","Lowest Price"]

plt.barh(L_max_y,H_max_x)
plt.title("TCS Stock -  Maximum Highest and Lowest Price",size=25)
 
for index, value in enumerate(H_max_x):
    plt.text(value, index,
             str(value))


# ## High vs Low - TCS

# In[28]:


plt.style.use("dark_background")
h_l_tcs=pd.DataFrame()
h_l_tcs["tcs_high"]=pd.Series(df.High)
h_l_tcs["tcs_low"]=pd.Series(df.Low)
h_l_tcs["Date"]=df.Date
h_l_tcs=h_l_tcs.set_index(["Date"])
h_l_tcs.iplot(kind="bar",color=["green","lightgreen"])


# ## Preprocessing

# In[29]:


df_df=df.drop(["Symbol","Series","Prev Close","VWAP","Trades","Deliverablevolume","Deliverable","Turnover"],axis=1)
df_df


# In[30]:


df_close_sqrt=sqrt(df_df["Close"])
df_close_log=log(df_df["Close"])


# In[31]:


fig,ax=plt.subplots(2,3)
sns.histplot(df["Close"],ax=ax[0,0]) ; ax[0,0].set_title("Original")
#ax[1]=plt.plot(df1["Close"])
ax[1,0].plot(df_df.Close)
plt.rcParams["figure.figsize"]=(18,9)
sns.histplot(df_close_sqrt,ax=ax[0,1]) ; ax[0,1].set_title("sqrt")
ax[1,1].plot(df_close_sqrt)
sns.histplot(df_close_log,ax=ax[0,2]) ; ax[0,2].set_title("log")
ax[1,2].plot(df_close_log)


# In[32]:


# dicky fuller test
adft=adfuller(df_close_log)
output=pd.Series(adft[0:4],index=["t-value","p-value","no.of.lags","no.of.observations"])
output


# ## Model Building 

# In[33]:


train=df_close_log.iloc[0:int(len(df_close_log)*.80)]
test=df_close_log.iloc[int(len(df_close_log)*.80):int(len(df_close_log))]


# In[34]:


train.shape,test.shape


# In[35]:


# 1.Simple Exponential Smoothing
decimals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
best_one=[]
for i in range(len(decimals)):
    model=SimpleExpSmoothing(train).fit(smoothing_level=decimals[i])
    preds=model.predict(start=test.index[0],end=test.index[-1])
    best_one.append(mean_absolute_error(preds,test))
for j in range(len(decimals)):
    print(decimals[j],":",best_one[j])


# In[36]:


# 2. Simple Exponential Smoothing
df_model1=SimpleExpSmoothing(train).fit(smoothing_level=1.0)
df_preds1=df_model1.predict(start=test.index[0],end=test.index[-1])
df_error1=mean_absolute_error(df_preds1,test)
df_error1


# In[37]:


# 2. Advance Exponential Smoothing ( HOLT )
decimals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
best_one=[]
for i in range(len(decimals)):
    model=Holt(train).fit(smoothing_level=1.0,smoothing_trend=decimals[i])
    preds=model.predict(start=test.index[0],end=test.index[-1])
    best_one.append(mean_absolute_error(preds,test))
for j in range(len(decimals)):
    print(decimals[j],":",best_one[j])


# In[38]:


# 2. Advance Exponential Smoothing ( HOLT )
df_model2=Holt(train).fit(smoothing_level=1.0,smoothing_trend=0.1)
df_preds2=df_model2.predict(start=test.index[0],end=test.index[-1])
df_error2=mean_absolute_error(df_preds2,test)
df_error2


# In[39]:


# 3. Holt's winter additive trend and aditive sesonality
df_model3=ExponentialSmoothing(train,trend="add",seasonal="add",seasonal_periods=12).fit()
df_preds3=df_model3.predict(start=test.index[0],end=test.index[-1])
df_error3=mean_absolute_error(df_preds3,test)
df_error3


# In[40]:


# 4. Holt's winter additive trend and multiplicative sesonality
df_model4=ExponentialSmoothing(train,trend="add",seasonal="mul",seasonal_periods=12).fit()
df_preds4=df_model4.predict(start=test.index[0],end=test.index[-1])
df_error4=mean_absolute_error(df_preds4,test)
df_error4


# In[41]:


# 6. Auto Regressive
df_model5 = AutoReg(train, lags=10).fit()
print(df_model5.summary())


# In[42]:


len(train)


# In[43]:


df_preds5 = df_model5.predict(start=len(train), end=2474, dynamic=False)
df_error5=mean_absolute_error(df_preds5,test)
df_error5


# In[44]:


test.shape


# In[45]:


# 6. ARIMA Method
df_model6 = ARIMA(train, order=(3,1,0))
df_model6 = df_model6.fit()
a=df_model6.forecast(495)
d=pd.Series(a,index=test.index)
df_error6=mean_absolute_error(d,test)
df_error6


# In[46]:


# 8. LSTM

#df_lstm=pd.DataFrame()
#df_lstm["1_day_back_price"]=df_close_log.shift(1)
#df_lstm["2_day_back_price"]=df_close_log.shift(2)
#df_lstm["3_day_back_price"]=df_close_log.shift(3)
#df_lstm["close"]=df_close_log
#df_lstm["Date"]=df_df["Date"]
#df_lstm=df_lstm.set_index(["Date"])
#df_lstm=df_lstm.dropna()
#df_lstm_x=df_lstm.drop(["close"],axis=1)
#df_lstm_y=df_lstm[["close"]]
#df_lstm_close=np.array(df_close_log).reshape(-1,1)

#df_lstm_x=np.array(df_lstm_x).reshape(-1,1)
#df_lstm_y=np.array(df_lstm_y).reshape(-1,1)

#df_lstm_xtrain=df_lstm_x[:1980]
#df_lstm_xtest=df_lstm_x[1980:]
#df_lstm_ytrain=df_lstm_y[:1980]
#df_lstm_ytest=df_lstm_y[1980:]

normalizing=MinMaxScaler(feature_range=(0,1))
df_lstm=normalizing.fit_transform(np.array(df_close_log).reshape(-1,1))

train_data=int(len(df_lstm)*0.80)
test_data=len(df_lstm)-train_data
train_data,test_data


# In[47]:


df_lstm_train,df_lstm_test=df_lstm[0:train_data,:],df_lstm[train_data:len(df_lstm),:]


# In[48]:


len(df_lstm_train),len(df_lstm_test)


# In[49]:


def create_df(dataset,step):
    xxtrain,yytrain=[],[]
    for i in range(len(dataset)-step-1):
        a=dataset[i:(i+step),0]
        xxtrain.append(a)
        yytrain.append(dataset[i+step,0])
    return np.array(xxtrain),np.array(yytrain)    


# In[50]:


t_s=100
df_lstm_xtrain,df_lstm_ytrain=(create_df(df_lstm_train,t_s))
df_lstm_xtest,df_lstm_ytest=(create_df(df_lstm_test,t_s))


# In[51]:


df_lstm_xtrain.shape,df_lstm_ytrain.shape,df_lstm_xtest.shape,df_lstm_ytest.shape


# In[52]:


n_features=1
df_lstm_xtrain=df_lstm_xtrain.reshape(df_lstm_xtrain.shape[0],df_lstm_xtrain.shape[1],n_features)
df_lstm_xtest=df_lstm_xtest.reshape(df_lstm_xtest.shape[0],df_lstm_xtest.shape[1],n_features)


# In[53]:


df_model7 = Sequential()
df_model7.add(LSTM(50,return_sequences=True,input_shape=(df_lstm_xtrain.shape[1],1)))
df_model7.add(LSTM(50,return_sequences=True))
df_model7.add(LSTM(50))
df_model7.add(Dense(1,activation='linear')) 
df_model7.compile(optimizer="adam",loss="MAE")
df_model7.fit(df_lstm_xtrain,df_lstm_ytrain,validation_data=(df_lstm_xtest,df_lstm_ytest),epochs=10,batch_size=64)


# In[54]:


df_preds7=df_model7.predict(df_lstm_xtest)
df_error7=mean_absolute_error(df_preds7,df_lstm_ytest)
df_error7


# In[55]:


plt.plot(df_preds7)
plt.plot(df_lstm_ytest)


# In[56]:


df_lstm_ytest.shape


# In[57]:


fut_inp=df_lstm_ytest[294:]
fut_inp=fut_inp.reshape(1,-1)


# In[58]:


temp_inp=list(fut_inp)
fut_inp.shape


# In[59]:


temp_inp=temp_inp[0].tolist()


# In[60]:


#n=int(input("Enter the number"))
lst_out=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_inp)>100):
        fut_inp=np.array(temp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp=fut_inp.reshape((1,n_steps,1))
        yhat=df_model7.predict(fut_inp,verbose=0)
        temp_inp.extend(yhat[0].tolist())
        temp_inp=temp_inp[1:]
        lst_out.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp=fut_inp.reshape((1,n_steps,1))
        yhat=df_model7.predict(fut_inp,verbose=0)
        temp_inp.extend(yhat[0].tolist())
        lst_out.extend(yhat.tolist())
        i=i+1
               
    print(lst_out) 


# In[61]:


np.exp(normalizing.inverse_transform(lst_out))


# In[62]:


plot_new=np.arange(1,101)
plot_pred=np.arange(101,131)


# In[63]:


df_lstm.shape,len(lst_out)


# In[64]:


plt.plot(plot_new,normalizing.inverse_transform(np.exp(df_lstm[2375:])))
plt.plot(plot_pred,normalizing.inverse_transform(np.exp(lst_out)))


# In[65]:


ranges1=np.arange(2081,2475)
ranges2=np.arange(2475,2505)
t1=np.exp(normalizing.inverse_transform(df_lstm))
t2=np.exp(normalizing.inverse_transform(df_model7.predict(df_lstm_xtrain)))
t3=np.exp(normalizing.inverse_transform(df_preds7))
t4=np.exp(normalizing.inverse_transform(lst_out))
plt.plot(t1)
plt.plot(t2)
plt.plot(ranges1,t3)
plt.plot(ranges2,t4)
label=["close","train","test","forcast"]
plt.legend(label)


# In[66]:


df_models=dict()
df_models["SES"]=1-df_error1
df_models["AES (HOLT'S)"]=1-df_error2
df_models["AES add(trend,sesonal)"]=1-df_error3
df_models["AES add(trend),mul(sesonal)"]=1-df_error4
df_models["AutoRegressive"]=1-df_error5
df_models["ARIMA"]=1-df_error6
df_models["LSTM"]=1-df_error7


# In[67]:


df_models=pd.DataFrame(list(df_models.items()),columns=["Model","Accuracy"])
df_models


# In[68]:


df_models=df_models.sort_values("Accuracy",ascending=False)
df_models


# In[69]:


sns.barplot(x="Accuracy",y="Model",data=df_models)


# ## Finalize Model for TCS is ( LSTM ) - supply entire data init

# In[70]:


df_x,df_y=create_df(df_lstm,t_s)
df_x.shape,df_y.shape


# In[71]:


df_x=df_x.reshape(df_x.shape[0],df_x.shape[1],n_features)


# In[72]:


df_model = Sequential()
df_model.add(LSTM(50,return_sequences=True,input_shape=(df_x.shape[1],1)))
df_model.add(LSTM(50,return_sequences=True))
df_model.add(LSTM(50))
df_model.add(Dense(1,activation='linear')) 
df_model.compile(optimizer="adam",loss="MAE")
df_model.fit(df_x,df_y,epochs=10,batch_size=64)


# In[73]:


df_y.shape


# In[74]:


fut_inp_=df_y[2274:]
fut_inp_=fut_inp_.reshape(1,-1)
temp_inp_=list(fut_inp_)
fut_inp_.shape


# In[75]:


temp_inp_=temp_inp_[0].tolist()


# In[76]:


lst_out_=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_inp_)>100):
        fut_inp_=np.array(temp_inp_[1:])
        fut_inp_=fut_inp_.reshape(1,-1)
        fut_inp_=fut_inp_.reshape((1,n_steps,1))
        yhat=df_model.predict(fut_inp_,verbose=0)
        temp_inp_.extend(yhat[0].tolist())
        temp_inp_=temp_inp_[1:]
        lst_out_.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp_=fut_inp_.reshape((1,n_steps,1))
        yhat=df_model.predict(fut_inp_,verbose=0)
        temp_inp_.extend(yhat[0].tolist())
        lst_out_.extend(yhat.tolist())
        i=i+1
               
    print(lst_out_) 


# In[77]:


np.exp(normalizing.inverse_transform(lst_out_))


# In[78]:


df_entire_data=df_lstm.tolist()
df_entire_data.extend(lst_out_)


# In[79]:


len(df_entire_data)


# In[80]:


plt.plot(df_entire_data)


# ## Generating file for Deployment

# In[82]:


df_model.save("tcs_model2.h5")


# In[ ]:




