import yfinance as yf 
import pandas
import matplotlib.pyplot as plt #for graphing
from sklearn.ensemble import RandomForestClassifier      
#  creates a set of decision trees from a randomly 
#  selected subset of the training set. It is basically a set of decision trees (DT) from a 
# randomly selected subset of the training set and then It collects the votes from different decision trees to decide the final prediction.
#  can pick up non lin relationships


# intiliaze ticker object to download price history for SP500


sp500 = yf.Ticker("^GSPC")

 #query historical prices from the very beginning until now using pandas dataframe
sp500 = sp500.history(period='max')
# print(sp500)


sp500.plot.line(y="Close", use_index = True) # uses dataframe plot.line for a line plot. Y AXIS = CLOSING PRICES , X-AXIS = YEARS 

del sp500['Dividends']
del sp500['Stock Splits']             #deletes unneccesary columns on DF 

sp500['Tomorrow'] = sp500['Close'].shift(-1)   # create new col TOMORROW = the CLOSE column (closing price of the following day)


# create another col TARGET = compares if the new column TOMORROW =
# (if closing price tomorrow is greater than todays closing) --> 1 if TRUE 0 if FALSE
sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)         



sp500 = sp500.loc['1990-01-01':].copy()   #PANDAS LOC ONLY TAKES DATA FROM 1990-NOW

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)   
#nestimators = num of individual decision trees to be trained and higher the more accurate.

# minsamples = prevents overfitting, higher the less accurate. sets the minimum number of samples that 
# must be present in a node for the algorithm to consider splitting it further. 
# generalizes better to not capute "noise"

# randstate = running mult times , random num will be in a predictable sequen (SAME RESULTS) good for seeing if u improved. 
# the random number generator produces the same results each time the code is run.

train = sp500.iloc[:-100]   #split into test train , time test data , cant use cross validation, 
# everything except the last hundred rows
test = sp500.iloc[-100:]  # last hundred into test set 

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']  # DONT USE TOMORROW OR TARGET COLUMNS, DOESNT WORK IN REAL WORLD

trainm = model.fit(train[predictors], train['Target'])  # supervised TRAINING

# use predictor columns inside the TRAIN , and out into the model which will do the decision trees 
# fit() passes predictor columns as input and TARGET is the output variable that the model will learn to predict
# print(trainm)      WILL OUTPUT RandomForestClassifier(min_samples_split=100, random_state=1)

#IMPORTANT TO MEASURE UR MODEL   The fit() method takes the training data as arguments, 
# which can be one array in the case of unsupervised learning (discover patterns, structures, or 
# relationships in the data without explicit guidance or labels to find meaningful clusters, 
# associations, or representations.), or two arrays in the case of supervised learning 
# Essentially, predict() will perform a prediction for each test instance 


from sklearn.metrics import precision_score  # calculates how much percent of the time its correct

preds = model.predict(test[predictors])   # selects columns as input into model , and predicts output

preds = pandas.Series(preds, index=test.index)   #numpy is too difficult, turn into pandas , and index using same test columns that was in preds 

 
prec_score = precision_score(test['Target'], preds)  # precision score of the Target values (sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)) 
#                                                      vs Prediction values


combine = pandas.concat([test['Target'], preds], axis = 1)   #concatenate our actual values and our predicted values , axis means treat each input as a column
combine.plot() 



# FUNCTIONS


def predict(train, test, predictors, model):
    #fitting the model, training predictions 
    model.fit(train[predictors], train['Target'])
    
    # predicting the last hundred days
    preds = model.predict(test[predictors])
    
    preds = pandas.Series(preds, index=test.index, name='Predictions')  
    combine = pandas.concat([test['Target'], preds], axis = 1)   
    #concatenate our actual values and our predicted values , axis means treat each input as a column
    
    return combine



def backtest(data, model, predictors, start=2500, step=250): 
    #  when u backtest u want a certain amount of data to train ur first model,
    #  every trading year has 250 days,  250 x 10 == 2500 , step means it will the train the model for a year 
    # predict the 11th year then the 12 then the 13
    all_preds = []
    
    for i in range(start, data.shape[0], step): #loops around all data year by year
        train = data.iloc[0:i].copy()  #split up training and test data  , training set is all the years prior to current year , and test set is current year
        
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_preds.append(predictions)
    return pandas.concat(all_preds) #

predictions = backtest(sp500, model, predictors)



predictions['Predictions'].value_counts()  #tells how many days we predict market goes up and down 


precisionscore = precision_score(predictions['Target'], predictions['Predictions']) 
#percent if the market goes up or down and my predictions compared to what actually happened





# ADD MORE PREDICTORS

horizon = [2, 5, 60, 250, 1000]  # timeline of days
new_preds = []
for horizon in horizon:
    rolling_avg = sp500.rolling(horizon).mean()   #loop thru horizon values and calc a rolling avg of those past days
    #                                              and take the mean of the closing ratio
    ratio_col = f'Close_Ratio_{horizon}' 
    
    sp500[ratio_col] = sp500['Close'] / rolling_avg['Close']   
    # ratio between each days close and the mean in the last 2 days, 5 days, 60 days, 250, and 1000
    # setting up a general model for prediction
    
    trend_col = f'Trend_{horizon}'
    sp500[trend_col] = sp500.shift(1).rolling(horizon).sum()['Target']   
    #on a given day looks at the past few days and see the avg sum of the target when it went up
    new_preds += [ratio_col, trend_col]       # new column added NaN meands not availabe, not enuf days
    
sp500 = sp500.dropna()    # drops missing rows


# IMPROVING MORE
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    #fitting the model, generating predictions 
    model.fit(train[predictors], train['Target'])
    # predicting the last hundred days
    preds = model.predict_proba(test[predictors])[:,1]   # PROBABILITY
    preds[preds >= .6 ] = 1
    preds[preds < .6] = 0   # if above 60 percent it is true   REDUCES TOTAL NUM OF TRADING DAYS
    
    preds = pandas.Series(preds, index=test.index, name='Predictions')  
    combine = pandas.concat([test['Target'], preds], axis = 1)   
    #concatenate our actual values and our predicted values , axis means treat each input as a column
    return combine

predictions = backtest(sp500, model, new_preds)   # not using close or open 


print(precision_score(predictions['Target'], predictions['Predictions']))   # percent of predictions vs actuality 
plt.show()
