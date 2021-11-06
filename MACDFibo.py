
#Fibonacci retracement levels and MACD to indicate when to buy and sold

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


#Get and show the data
df = pd.read_csv('./newbinance.csv')
df = df.set_index(pd.DatetimeIndex(df['date'].values))
df = df.iloc[::-1]
df = df[-300:]
#df

#Plot the data
plt.figure(figsize=(14,7))
plt.plot(df.close)
plt.title('BTC/USD Close price')
plt.xlabel('Date')
plt.ylabel('Price ($USD)')
plt.xticks(rotation=0)
plt.show()

#Calculate fibonacci retracement levels
max_price = df['close'].max()
min_price = df['close'].min()

difference = max_price - min_price
first_level = max_price - difference * 0.236
second_level = max_price - difference * 0.382
third_level = max_price - difference * 0.5
four_level = max_price - difference * 0.618

#Calculate the MACD line and the signal line indicators
#Calculate the Short Term Exponential Moving Average
ShortEMA = df.close.ewm(span=12, adjust=False).mean()

#Calculate the Long Term Exponential Moving Average
LongEMA = df.close.ewm(span=26, adjust=False).mean()

#Calculate the Moving Average Convergence/Divergence (MACD)
MACD = ShortEMA - LongEMA

#Calculate the signal line
signal = MACD.ewm(span=9, adjust=False).mean()

#Plot indicators
new_df = df

#plot fibonacci
plt.figure(figsize=(12.33, 10))
plt.subplot(2,1,1)
plt.plot(new_df.index, new_df['close'])
plt.axhline(max_price, linestyle='--', alpha=0.5, color='red')
plt.axhline(first_level, linestyle='--', alpha=0.5, color='orange')
plt.axhline(second_level, linestyle='--', alpha=0.5, color='yellow')
plt.axhline(third_level, linestyle='--', alpha=0.5, color='green')
plt.axhline(four_level, linestyle='--', alpha=0.5, color='blue')
plt.axhline(min_price, linestyle='--', alpha=0.5, color='purple')
plt.ylabel('Fibonacci')
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)


#Plot MACD and Signal Line
plt.subplot(2,1,2)
plt.plot(new_df.index, MACD)
plt.plot(new_df.index, signal)
plt.ylabel('MACD')
plt.xticks(rotation=0)

plt.savefig('Fig1.png')

#Create new columns for the df
df['MACD'] = MACD
df['Signal Line'] = signal
#Show the new data
df

#Create a function to be used in our strategy to get the upper Fibonnaci level and the lower fibbo level of the current price
def getLevels(price):
  if price >= first_level:
    return (max_price, first_level)
  elif price >= second_level:
    return (first_level, second_level)
  elif price >= third_level:
    return (second_level, third_level)
  elif price >= four_level :
    return (third_level, four_level)
  else: 
    return (four_level, min_price)

#Create a function for the trading startegy

#The strategy :
#When the signal line crosses above the MACD Line and the current price crossed above or below the last fibo level then buy
#If the signal line crosses below the MACD Line and the current price crossed above or below the last fibo level then sell

def startegy(df):
  buy_list = []
  sell_list = []
  flag = 0
  last_buy_price = 0
  new_stop_loss = 0
  max_loss = -0.07

  #Loop throught the data set
  for i in range(0, df.shape[0]):
    price = df['close'][i]
    #if this is the first data point whithin the data set, then get the level above and below it.
    if i == 0:
      upper_lvl, lower_lvl = getLevels(price)
      buy_list.append(np.nan)
      sell_list.append(np.nan)
    #Set StopLoss at max-loss
    elif ((price - new_stop_loss) / new_stop_loss) < max_loss and flag == 1: 
      buy_list.append(np.nan)
      sell_list.append(price)
      #Set flag to 0 to signal the position was sold
      flag = 0
    #Take proffit 10% Up
    elif ((price - last_buy_price) / last_buy_price) > 0.1 and flag == 1 and price > first_level: 
      buy_list.append(np.nan)
      sell_list.append(price)
      #Set flag to 0 to signal the position was sold
      flag = 0    

    #Else if the current price is greater than or equal to the upper_lvl or less than or equal to the lower_lvl, then we know the price has 'hit' or crossed fibbo level
    elif price >= upper_lvl or price <= lower_lvl:

      #Check to see if the MACD line crossed above or below the signal line
      if df['Signal Line'][i] > df['MACD'][i] and flag == 0:
        last_buy_price = price
        buy_list.append(price)
        sell_list.append(np.nan)
        #Set the flag to 1 to signal the position taked
        flag = 1
        #Set the stopLoss price 0
        new_stop_loss = price

      elif df['Signal Line'][i] < df['MACD'][i] and flag == 1 and price >= last_buy_price:
        buy_list.append(np.nan)
        sell_list.append(price)
        #Set flag to 0 to signal the position was sold
        flag = 0
      else: 
        buy_list.append(np.nan)
        sell_list.append(np.nan)

    else: 
      buy_list.append(np.nan)
      sell_list.append(np.nan)

    #Update the new levels
    upper_lvl, lower_lvl = getLevels(price)
  
  return buy_list, sell_list

#Create buy and sell columns
buy, sell = startegy(df)
df['Buy_Signal_Price'] = buy
df['Sell_Signal_Price'] = sell


#Plot fibo levels along whit the close price and whit the buy and sell signals
#Plot indicators
new_df = df.copy()

#plot fibonacci
plt.figure(figsize=(12.33, 10))
plt.subplot(2,1,1)
plt.plot(new_df.index, new_df['close'], alpha = 0.5)
plt.scatter(new_df.index, new_df['Buy_Signal_Price'], color='green', marker='^')
plt.scatter(new_df.index, new_df['Sell_Signal_Price'], color='red', marker='v')
plt.axhline(max_price, linestyle='--', alpha=0.5, color='red')
plt.axhline(first_level, linestyle='--', alpha=0.5, color='orange')
plt.axhline(second_level, linestyle='--', alpha=0.5, color='yellow')
plt.axhline(third_level, linestyle='--', alpha=0.5, color='green')
plt.axhline(four_level, linestyle='--', alpha=0.5, color='blue')
plt.axhline(min_price, linestyle='--', alpha=0.5, color='purple')
plt.ylabel('Close Price in USD')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.show()

#Calculate profit
data_buy = df['Buy_Signal_Price'][df['Buy_Signal_Price']> 0]
data_sell = df['Sell_Signal_Price'][df['Sell_Signal_Price']> 0]
buy_dates = []
sell_dates = []
days = []
profit = []
acumulated = 0

for i in range(0, data_buy.shape[0]):
  buy_dates.append(data_buy.index[i])
  sell_dates.append(data_sell.index[i])
  days.append(data_sell.index[i] - data_buy.index[i])
  acumulated = acumulated + (data_sell[i] - data_buy[i]) / data_buy[i]
  difference = "{:.2%}".format((data_sell[i] - data_buy[i]) / data_buy[i] )
  profit.append(difference)

data = {'time_difference': days ,'profit': profit, 'buy_time': buy_dates, 'sell_time': sell_dates }
df_profits = pd.DataFrame(data)

acumulated = "{:.2%}".format(acumulated)

print(df_profits)
print(acumulated)