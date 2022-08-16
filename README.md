# ROBO-ADVISOR from scratch


## Motivation
 
The risk tolerance of an investor is one of the most important inputs in the robo-advisor workflow process. This robo-advisor is going to use a wide variety of risk profiling tools to measure the risk tolerance of an investor. The focus is on a regression-based algorithm to compute an investor’s risk tolerance and automate manual processes involved in the client screening process, followed by demonstrating the model graphically. Risk tolerance will be predicted by a Machine Learning model. We’ll also consider whether machine learning models might be able to objectively analyze the behavior of investors in a changing market and attribute these changes to the investor’s risk appetite using automated dashboard.


## Data collecting and preparation 

Our target values are the “true” risk tolerance of an individual and the features are demographic, financial, and behavioral attributes of a prospective investor. We have 19,285 observations with 515 columns.

The data source: [Survey of Consumer Finances (SCF)]( https://www.federalreserve.gov/econres/scf_2009p.htm). The historical data includes responses about household demographics, net worth, financial, and nonfinancial assets for the same set of individuals from 2007 (pre-crisis) to 2009 (post-crisis). 

## Model training

We first normalize the risky assets with the price of a stock index (S&P500) in 2007 versus 2009 to get risk tolerance by taking the ratio of risky assets to total assets of an individual. 
Looking at the risk tolerance of 2007, we see that a significant number of individuals had a risk tolerance close to 1. This means that their investments were more skewed towards the risky assets as compared to the riskless assets:

<img width="439" alt="2007" src="https://user-images.githubusercontent.com/53462948/184729355-9afa5b74-2ab0-448c-b1c9-e6ff59318f61.png">

The situation reversed in 2009 after the crisis and majority of the investment was in risk free assets. Overall risk tolerance decreased, which is shown by majority of risk tolerance being close to 0 in 2009: 

<img width="413" alt="2019" src="https://user-images.githubusercontent.com/53462948/184729465-cc5b47ca-6800-4819-876b-81ee416d28ef.png">

We assign the true risk tolerance as the average risk tolerance of these savvy investors between 2007 and 2009. This is the predicted variable for the modeling in this part of the robo construction process. The goal is to predict the true risk tolerance of an individual given the demographic, financial, and willingness to take risk, features.

To filter the variables, we will check the description in the data dictionary and keep only the features that are relevant. Looking at the entire data set, there are more than 500 features in the dataset. However, the analysis indicates that the risk tolerance is heavily influenced by investor demographic, financial, and behavioral attributes, such as age, current income, net worth, and willingness to take risk. All these attributes are available in the dataset. These attributes are used as features to predict investors’ risk tolerance. In the dataset, each of the columns contains a numeric value corresponding to the value of the attribute.

We will, next, evaluate the correlations between the variables and risk tolerance. We can notice from the Seaborn correlation heatmap that there is positive correlation with all parameters except the number of kids and marital status. 

<img width="748" alt="sns" src="https://user-images.githubusercontent.com/53462948/184735279-3a4121f7-96c1-4d47-9ece-67bb878ee737.png">

## Modeling

We used a decision tree model a baseline one and then optimized the hyperparameters of random forest model for achieving the minimum MSE (mean squared error). It turns out that random forest model could handle the situation well enough with the tuned hyperparameters. The final R2 was 0.8 while MSE = 0.006 providing sufficient results to launch our app.

## Main logic

The app allows users to pick their desired stocks and then they are prompted to adjust their parameters such as age, marital status, net worth, income, education level, number of kids, occupation and the willingness to take a risk for understanding the risk tolerance level. The users are also provided by the expected asset allocation. We have used Dash Python for establishing the interactive app.


## Collaboration

Please, feel free to contribute if you're interested. Commands to run for getting started:


- Install the required packages
```
make install
```

- Run the app

```
python main.py
```

## Demo

<img width="1509" alt="demo" src="https://user-images.githubusercontent.com/53462948/184764367-4f5ee7b3-b6b9-419e-869d-e420798355b3.png">

