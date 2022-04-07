import dash
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import cvxopt as opt
from cvxopt import solvers
import logging
logging.getLogger("requests").setLevel(logging.WARNING)
import dash_core_components as dcc
import dash_html_components as html

#added comment


investors = pd.read_csv('./data/dataset.csv', index_col = 0)
assets = pd.read_csv('./data/SP500Data.csv',index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
assets=assets.fillna(method='ffill')
options=np.array(assets.columns)

options = []
for tic in assets.columns:
    mydict = {}
    mydict['label'] = tic 
    mydict['value'] = tic
    options.append(mydict)


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([ 
        #Dashboard Name
        html.Div([
            html.Div([
                html.H3(children='Robo-Advisor'),
            ], style={'background-color': 'blue'}),
            html.Div([
                html.H5(children='Step 1 : Investor Characteristics '),
                html.Div([
                html.Label('Age', style={'padding': 10}),
                dcc.Slider(
                    id='Age',
                    min=investors['AGE07'].min(),
                    max=70,
                    marks={25: '25', 35: '35', 45: '45', 55: '55', 70: '70'},
                    value=25),
                html.Br(),

                html.Label('Net Worth ($Millions)', style={'padding': 10}),
                dcc.Slider(
                    id='Nwcat',
                    min=-1000000, max=3000000,
                    marks={-1000000: '-1.0', 0: '0.0', 500000: '0.5', 1000000: '1.0', 2000000: '2.0', },
                    value=10000),
                html.Br(),
                html.Label('Income ($Millions)', style={'padding': 10}),
                dcc.Slider(
                    id='Inccl',
                    min=-1000000,
                    max=3000000,
                    marks={-1000000: '-1.0', 0: '0.0', 500000: '0.5', 1000000: '1.0', 2000000: '2.0', },
                    value=100000),

                html.Br(),
                html.Label('Education Level', style={'padding': 10}),
                dcc.Slider(
                    id='Edu',
                    min=investors['EDCL07'].min(), max=investors['EDCL07'].max(),
                    marks={1: 'No High School', 2: '2', 3: '3', 4: 'College Degree'},
                    value=2),
                html.Br(),
                html.Label('Married', style={'padding': 10}),
                dcc.Slider(
                    id='Married',
                    min=investors['MARRIED07'].min(), max=investors['MARRIED07'].max(),
                    marks={1: 'Married', 2: 'Unmarried'},
                    value=1),
                html.Br(),
                html.Label('Kids', style={'padding': 10}),
                dcc.Slider(
                    id='Kids',
                    min=investors['KIDS07'].min(), max=investors['KIDS07'].max(),
                    # marks={ 1: '1',2: '2',3: '3',4: '4'},
                    marks=[{'label': j, 'value': j} for j in investors['KIDS07'].unique()],
                    value=3),
                html.Br(),
                html.Label('Occupation', style={'padding': 10}),
                dcc.Slider(
                    id='Occ',
                    min=investors['OCCAT107'].min(), max=investors['OCCAT107'].max(),
                    marks={1: 'Managerial', 2: '2', 3: '3', 4: 'Unemployed'},
                    value=3),
                html.Br(),
                html.Label('Willingness to take Risk', style={'padding': 10}),
                dcc.Slider(
                    id='Risk',
                    min=investors['RISK07'].min(), max=investors['RISK07'].max(),
                    marks={1: '1 (Highest)', 2: '2', 3: '3', 4: '4 (Lowest)'},
                    value=3),
                html.Br(),
                # html.Button(id='investor_char_button',
                #             n_clicks=0,
                #             children='Calculate Risk Tolerance',
                #             style={'fontSize': 14, 'marginLeft': '30px', 'color': 'white', \
                #                    'horizontal-align': 'left', 'backgroundColor': 'grey'}),
                # html.Br(),
                ], style={'width': '90%', 'text-align': 'center', 'display': 'inline-block'}),
                ], style={'display': 'inline-block','text-align': 'center', 'vertical-align': 'top',  'width': '30%',\
                   'color':'black', 'background-color': 'LightGray', 'border-radius': '25px'}),
            html.Div([
                html.H5(children='Step 2 : Portfolio management'),

                html.Div([
                # html.H5(children='Step 2 : Enter the Instruments for the allocation portfolio'),
                 html.Div([
                    html.Div([
                        html.Label('Risk Tolerance (scale of 100) :', style={'padding': 5}),
                        dcc.Input(id='risk-tolerance-text'),

                    ], style={'width': '100%', 'font-family': 'calibri', 'vertical-align': 'top',
                              'display': 'inline-block'}),

                    html.Div([
                        html.Label('Select the assets for the portfolio:', style={'padding': 5}),
                        dcc.Dropdown(
                            id='ticker_symbol',
                            options=options,
                            value=['GOOGL', 'FB', 'GS', 'MS', 'GE', 'MSFT'],
                            multi=True
                            # style={'fontSize': 24, 'width': 75}
                        ),
                        html.Button(id='submit-asset_alloc_button',
                                    n_clicks=0,
                                    children='Submit',
                                    style={'fontSize': 12, 'marginLeft': '25px', 'color': 'white',
                                           'backgroundColor': 'grey'}

                                    ),
                    ], style={'width': '80%', 'font-family': 'calibri', 'vertical-align': 'top',
                              'display': 'inline-block'}),
                 ], style={'width': '100%', 'display': 'inline-block', 'font-family': 'calibri', 'vertical-align': 'top'}),

                 html.Div([
                    html.Div([
                        dcc.Graph(id='Asset-Allocation'),
                    ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block',
                              'font-family': 'calibri', 'horizontal-align': 'right'}),
                    html.Div([
                        dcc.Graph(id='Performance')
                    ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block',
                              'font-family': 'calibri', 'horizontal-align': 'right'}),
                 ], style={'width': '100%', 'vertical-align': 'top', 'display': 'inline-block',
                          'font-family': 'calibri', 'horizontal-align': 'right'}),

                ], style={'width': '100%', 'display': 'inline-block', 'font-family': 'calibri', 'vertical-align': 'top',
                      'horizontal-align': 'right'}),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                         'color':'white','horizontalAlign' : "left", 'width': '70%', 'background-color':'blue',
                         'border-radius': '25px'}),

        ], style={'width': '100%', 'display': 'inline-block', 'font-family': 'calibri', 'vertical-align': 'top'}),
            ],style={'font-family': 'calibri', 'text-align': 'center'}),
         
         #All the Investor Characteristics
                      
         #html.Div([
          # html.Div([
          #
          #   html.Label('Age:',style={'padding': 5}),
          #   dcc.Slider(
          #       id='Age',
          #       min = investors['AGE07'].min(),
          #       max = 70,
          #       marks={ 25: '25',35: '35',45: '45',55: '55',70: '70'},
          #       value=25),
          #   html.Br(),
          #
          #   html.Label('NetWorth:', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Nwcat',
          #       min = -1000000, max = 3000000,
          #       marks={-1000000: '-$1M',0: '0',500000: '$0.5m',1000000: '$1M',2000000: '$2M',},
          #       value=10000),
          #   html.Br(),
          #   html.Label('Income:', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Inccl',
          #       min = -1000000,
          #       max = 3000000,
          #       marks={-1000000: '-$1M',0: '0',500000: '$500K',1000000: '$1M',2000000: '$2M',},
          #       value=100000),
          #
          #   html.Br(),
          #   html.Label('Education Level (scale of 4):', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Edu',
          #       min = investors['EDCL07'].min(), max = investors['EDCL07'].max(),
          #       marks={ 1: '1',2: '2',3: '3',4: '4'},
          #       value=2),
          #   html.Br(),
          #   html.Label('Married:', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Married',
          #       min = investors['MARRIED07'].min(), max = investors['MARRIED07'].max(),
          #       marks={ 1: '1',2: '2'},
          #       value=1),
          #   html.Br(),
          #   html.Label('Kids:', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Kids',
          #       min = investors['KIDS07'].min(), max = investors['KIDS07'].max(),
          #       #marks={ 1: '1',2: '2',3: '3',4: '4'},
          #       marks=[{'label': j, 'value': j} for j in investors['KIDS07'].unique()],
          #       value=3),
          #   html.Br(),
          #   html.Label('Occupation:', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Occ',
          #       min = investors['OCCAT107'].min(), max = investors['OCCAT107'].max(),
          #       marks={ 1: '1',2: '2',3: '3',4: '4'},
          #       value=3),
          #   html.Br(),
          #   html.Label('Willingness to take Risk:', style={'padding': 5}),
          #   dcc.Slider(
          #       id='Risk',
          #       min = investors['RISK07'].min(), max = investors['RISK07'].max(),
          #       marks={ 1: '1',2: '2',3: '3',4: '4'},
          #       value=3),
          #   html.Br(),
          #   html.Button(id='investor_char_button',
          #                   n_clicks = 0,
          #                   children = 'Calculate Risk Tolerance',
          #                   style = {'fontSize': 14, 'marginLeft': '30px', 'color' : 'white',\
          #                            'horizontal-align': 'left','backgroundColor': 'grey'}),
          #   html.Br(),
          #     ],style={'width': '100%', 'background-color': 'LightGray', 'border-radius': '25px'}),
          #
            #],style={'width': '30%', 'font-family': 'calibri','vertical-align': 'top','display': 'inline-block'\
            #         }),

    # ********************Risk Tolerance Charts********            
    #      html.Div([
    #            #html.H5(children='Step 2 : Enter the Instruments for the allocation portfolio'),
    #       html.Div([
    #         html.Div([
    #             html.Label('Risk Tolerance (scale of 100) :', style={'padding': 5}),
    #             dcc.Input(id= 'risk-tolerance-text'),
    #
    #             ],style={'width': '100%','font-family': 'calibri','vertical-align': 'top','display': 'inline-block'}),
    #
    #         html.Div([
    #             html.Label('Select the assets for the portfolio:', style={'padding': 5}),
    #             dcc.Dropdown(
    #                     id='ticker_symbol',
    #                     options = options,
    #                     value = ['GOOGL', 'FB', 'GS','MS','GE','MSFT'],
    #                     multi = True
    #                     # style={'fontSize': 24, 'width': 75}
    #                     ),
    #             html.Button(id='submit-asset_alloc_button',
    #                         n_clicks = 0,
    #                         children = 'Submit',
    #                         style = {'fontSize': 12, 'marginLeft': '25px','color' : 'white', 'backgroundColor': 'grey'}
    #
    #             ),
    #            ],style={'width': '100%','font-family': 'calibri','vertical-align': 'top','display': 'inline-block'}),
    #         ],style={'width': '100%','display': 'inline-block','font-family': 'calibri','vertical-align': 'top'}),
    #
    #         html.Div([
    #             html.Div([
    #                 dcc.Graph(id='Asset-Allocation'),
    #                 ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block', \
    #                   'font-family': 'calibri', 'horizontal-align': 'right'}),
    #             html.Div([
    #                 dcc.Graph(id='Performance')
    #                 ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block', \
    #                   'font-family': 'calibri', 'horizontal-align': 'right'}),
    #                ], style={'width': '100%', 'vertical-align': 'top', 'display': 'inline-block', \
    #                       'font-family': 'calibri', 'horizontal-align': 'right'}),
    #
    #
    #     ], style={'width': '70%','display': 'inline-block','font-family': 'calibri','vertical-align': 'top', 'horizontal-align': 'right'}),
    #    ],style={'width': '100%','display': 'inline-block','font-family': 'calibri','vertical-align': 'top'}),

  ])    

def predict_riskTolerance(X_input):
    loaded_model = load(open('final_model.sav', 'rb'))
    # estimate accuracy on validation set
    predictions = loaded_model.predict(X_input)
    return predictions

#Asset allocation given the Return, variance
def get_asset_allocation(riskTolerance,stock_ticker):
    #ipdb.set_trace()
    assets_selected = assets.loc[:,stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1-riskTolerance
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    w=portfolios['x'].T
    print (w)
    Alloc =  pd.DataFrame(data = np.array(portfolios['x']),index = assets_selected.columns)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    returns_final=(np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final,axis =1)
    returns_sum_pd = pd.DataFrame(returns_sum, index = assets.index )
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + 100   
    return Alloc,returns_sum_pd



#Callback for the graph
#This function takes all the inputs and computes the cluster and the risk tolerance


@app.callback(
    [Output('risk-tolerance-text', 'value')],
    #[Input('investor_char_button', 'n_clicks'),
    [Input('Age', 'value'),Input('Nwcat', 'value'),
    Input('Inccl', 'value'), Input('Risk', 'value'),
    Input('Edu', 'value'),Input('Married', 'value'),
    Input('Kids', 'value'),Input('Occ', 'value')])
#get the x and y axis details 

def update_risk_tolerance(Age,Nwcat,Inccl,Risk,Edu,Married,Kids,Occ):
      
    #ipdb.set_trace()

    n_clicks = 0
    RiskTolerance = 0
    if n_clicks != None:    
        X_input = [[Age,Edu,Married,Kids,Occ,Inccl, Risk,Nwcat]]
        RiskTolerance= predict_riskTolerance(X_input)
    #print(RiskAversion)
    #Using linear regression to get the risk tolerance within the cluster.    
    return list([round(float(RiskTolerance*100),2)])

@app.callback([Output('Asset-Allocation', 'figure'),
              Output('Performance', 'figure')],
            [Input('submit-asset_alloc_button', 'n_clicks'),
            Input('risk-tolerance-text', 'value')], 
            [State('ticker_symbol', 'value')
            ])
def update_asset_allocationChart(n_clicks, risk_tolerance, stock_ticker):
    
    Allocated, InvestmentReturn = get_asset_allocation(risk_tolerance,stock_ticker)  
    
    return [{'data' : [go.Bar(
                        x=Allocated.index,
                        y=Allocated.iloc[:,0],
                        marker=dict(color='red'),
                    ),
                    ],
            'layout': {'title':" Asset allocation - Mean-Variance Allocation"}

       },
            {'data' : [go.Scatter(
                        x=InvestmentReturn.index,
                        y=InvestmentReturn.iloc[:,0],
                        name = 'OEE (%)',
                        marker=dict(color='red'),
                    ),
                    ],
            'layout': {'title':"Portfolio value of $100 investment"}

       }]

if __name__ == '__main__':
    app.run_server()