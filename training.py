import numpy as np
import pandas as pd
from pickle import dump
from sklearn.tree import DecisionTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
import copy
from sklearn.model_selection import train_test_split



df = pd.read_excel('./data/SCFP2009panel.xlsx')
warnings.filterwarnings('ignore')

Average_SP500_2007=1478
Average_SP500_2009=948


# Taking the ratio of risky assets to total assets of an individual and consider that as a measure of 
# the individual’s risk tolerance. We normalize the risky assets with the price of a stock index (S&P500) 
# in 2007 versus 2009 to get risk tolerance. 



# 2007 risk tolerance

df['RiskFree07']= df['LIQ07'] + df['CDS07'] + df['SAVBND07'] + df['CASHLI07']

df['Risky07'] = df['NMMF07'] + df['STOCKS07'] + df['BOND07'] 

df['RT07'] = df['Risky07']/(df['Risky07']+df['RiskFree07'])


# 2009 risk tolerance

df['RiskFree09']= df['LIQ09'] + df['CDS09'] + df['SAVBND09'] + df['CASHLI09']

df['Risky09'] = df['NMMF09'] + df['STOCKS09'] + df['BOND09'] 

df['RT09'] = df['Risky09']/(df['Risky09']+df['RiskFree09'])



df['RT09'] = df['RT09']*(Average_SP500_2009/Average_SP500_2007)



df2 = copy.deepcopy(df)  
df2.head()
df2['PercentageChange'] = np.abs(df2['RT09']/df2['RT07']-1)


# delete null values
df2=df2.dropna(axis=0)
df2=df2[~df2.isin([np.nan, np.inf, -np.inf]).any(1)]


# Risk tolerance diagrams


#sns.distplot(df2['RT07'], hist=True, kde=False, bins=int(180/5), color = 'blue', hist_kws={'edgecolor':'black'})
#sns.distplot(df2['RT09'], hist=True, kde=False, bins=int(180/5), color = 'blue', hist_kws={'edgecolor':'black'})



df3 = copy.deepcopy(df2)
df3 = df3[df3['PercentageChange']<=.1]


# We assign the true risk tolerance as the average risk tolerance of these savvy investors between 2007 and 2009. 
# This is the predicted variable for the modeling in this part of the robo construction process. 
# The goal is to predict the true risk tolerance of an individual given the demographic, financial, and willingness 
# to take risk, features.

df3['TrueRiskTolerance'] = (df3['RT07'] + df3['RT09'])/2
df3.drop(labels=['RT07', 'RT09'], axis=1, inplace=True)
df3.drop(labels=['PercentageChange'], axis=1, inplace=True)
keep_list2 = ['AGE07','EDCL07','MARRIED07','KIDS07','OCCAT107','INCOME07','RISK07','NETWORTH07','TrueRiskTolerance']
drop_list2 = [col for col in df3.columns if col not in keep_list2]
df3.drop(labels=drop_list2, axis=1, inplace=True)



# Saving the final dataset into csv file

df3.to_csv('./data/dataset.csv')





##Split data into training and test sets 

Y=df3["TrueRiskTolerance"]
X=df3.loc[:,df3.columns != "TrueRiskTolerance"]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=3)


# Baseline model - decision tree


# Train the tree model
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train,y_train)



tree_predictions = tree_model.predict(X_test)
tree_r2 = r2_score(y_test, tree_predictions)
tree_mse = mean_squared_error(y_test, tree_predictions)
print('Test set r2 is {:.3f}'.format(tree_r2))
print('Test set mse is {:.3f}'.format(tree_mse))


# Random Forest



# Grid search to find optimal Random Forest hyperparameters
params = {'min_samples_leaf':[1,5,20],'n_estimators':[100,1000],
          'max_features':[0.1,0.5,1.],'max_samples':[0.5,None]}

rf_model = RandomForestRegressor(random_state=3)
grid_search = GridSearchCV(rf_model,params,cv=3)
grid_search.fit(X_train,y_train)

# Display the best parameters
grid_search.best_params_


# Retrain the optimal model on full training set
best_rf_model = RandomForestRegressor(**grid_search.best_params_,random_state=3)
best_rf_model.fit(X_train,y_train)



rf_predictions = best_rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
print('Test set r2 is {:.3f}'.format(rf_r2))
print('Test set mse is {:.3f}'.format(rf_mse))



# Saving the model
filename = 'final_model.sav'
dump(best_rf_model, open(filename, 'wb'))
