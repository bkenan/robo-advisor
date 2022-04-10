# ROBO-ADVISOR from scratch


## Motivation
 
The risk tolerance of an investor is one of the most important inputs in the robo-advisor workflow process. This robo-advisor is going to use a wide variety of risk profiling tools to measure the risk tolerance of an investor. The focus is on a regression-based algorithm to compute an investor’s risk tolerance and automate manual processes involved in the client screening process, followed by demonstrating the model graphically.Risk tolerance will be predicted by Machine Learning model. We’ll also consider whether machine learning models might be able to objectively analyze the behavior of investors in a changing market and attribute these changes to the investor’s risk appetite using automated dashboard.


## Data collecting and preparation 

Our target values are the “true” risk tolerance of an individual and the features are demographic, financial, and behavioral attributes of a prospective investor. We have 19,285 observations with 515 columns.


The data source: [Survey of Consumer Finances (SCF)]( https://www.federalreserve.gov/econres/scf_2009p.htm). The survey includes responses about household demographics, net worth, financial, and nonfinancial assets for the same set of individuals in 2007 (pre-crisis) and 2009 (post-crisis). 


### The steps for installation:

1. Clone this repo to your local machine
2. $ pip install dash
3. $ python main.py
