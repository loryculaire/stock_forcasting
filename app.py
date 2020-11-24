# import necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import plotly.express as px

import numpy as np
import pandas as pd
import fbprophet
from fbprophet import Prophet
from pytickersymbols import PyTickerSymbols
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
import holidays
import datetime

from sklearn import datasets
from sklearn.cluster import KMeans


# parameters 

START_DATE="01/01/2018"
SPLIT_DATE = "2020-10-01"
YEARS=[2018,2019,2020]
TICKER_NAME = "HMI.F"
INDICES_NAMES = ['CAC 40', 'FTSE 100', 'NASDAQ 100', 'S&P 500', 'DAX', 'DOW JONES']
FILTER = 'sector'

# get list of tickers and meta data from indices 
def get_tickets(indices_names):
    
    stock_data = PyTickerSymbols()

    tickers = []
    for i in indices_names:
        tickers += list(stock_data.get_stocks_by_index(i))

    tickers = list(tickers)

    names = []
    yahoo_ticker = []
    industry = []
    sector = []
    country = []
    indices = []
    for i in range(len(tickers)):
        names.append(tickers[i]['name'])
        yahoo_ticker.append(tickers[i]['symbols'][0]['yahoo'])
        try:
            industry.append(tickers[i]["industries"][0])
        except:
            industry.append('NaN')  
        try:
            sector.append(tickers[i]["industries"][-1])
        except:
            sector.append('NaN')
        country.append(tickers[i]["country"])
        indices.append(tickers[i]["indices"])


    tickers_df = pd.DataFrame()
    tickers_df['name'] = names
    tickers_df['yahoo_ticker'] = yahoo_ticker
    tickers_df['industry'] = industry
    tickers_df['sector'] = sector
    tickers_df['country'] = country
    tickers_df['indices'] = indices

    tickers_df.index = yahoo_ticker
    tickers_df = tickers_df.drop_duplicates(subset=['yahoo_ticker'])
    return tickers_df


def get_holidays(tickers_df, ticker_name):
    
    country_to_code = {'Netherlands':'NLD', 'France':'FRA', 'Luxembourg':'LUX', 'Switzerland':'CHE',
       'United Kingdom':'GBR', 'Ireland':'IE', 'Russian Federation':'RUS', 'Mexico':'MEX',
       'Bermuda':'USA', 'United States':'USA', 'Germany':'DE', 'China':'USA', 'Israel':'ISR'}
    
    holidays_df = holidays.CountryHoliday(country_to_code[tickers_df.loc[ticker_name]['country']], 
                    years=YEARS)
    holidays_df = pd.DataFrame.from_dict(holidays_df, orient="index")
    holidays_df["ds"]=holidays_df.index
    holidays_df["holiday"]=holidays_df[0]
    del holidays_df[0]
    
    return holidays_df


def get_ticker_regressors(tickers_df, ticker_name, filter_name):

    regressor_tickers = tickers_df[tickers_df[filter_name] == tickers_df.loc[ticker_name][filter_name]].index

    tickers_data = {}
    for ticker in regressor_tickers:
        try:
            #print(f"fetching {ticker}")
            tickers_data[ticker] = {"data" : si.get_data(ticker, start_date=START_DATE, interval='1d'), 
                                    "industry": tickers_df.loc[ticker].industry, 
                                    "sector": tickers_df.loc[ticker].sector, 
                                    "country": tickers_df.loc[ticker].country, 
                                    "indices": tickers_df.loc[ticker].indices,
                                    "name": tickers_df.loc[ticker].name
                                    } 
        except:
            print(f"error on ticker {ticker}")
            pass     

    return tickers_data


# get finacial regressors

## create function for calculating profit margins

def margin(x, y):
  try:
    return(float(x)/float(y))
  except:
    return('NaN')

# create function for calculating stock growth and revenue growth

def per_growth(x, y):
  try:
    return (float(x) / float(y) - 1)
  except ZeroDivisionError:
    return('NaN')

def get_fin_regressors(ticker_code):
    # TEST : get financial data for one ticker
    data = si.get_financials(ticker=ticker_code)['yearly_income_statement'].transpose()
    data['co_name'] = tickers_df.loc[ticker_code]['name']
    data = data.sort_values(by='endDate')
    #data.loc['ds'] = data.loc['endDate']

    # remove duplicates

    data.drop_duplicates(subset=['totalRevenue'], inplace=True)
    data = data.reset_index()
    data.index = data.endDate

    # calculate aggregates

    data['grossProfit_margin'] = data.apply(lambda x : margin(x['grossProfit'],x['totalRevenue']), axis=1)
    data['operatingIncome_margin'] = data.apply(lambda x : margin(x['operatingIncome'],x['totalRevenue']), axis=1)
    data['netIncome_margin'] = data.apply(lambda x : margin(x['netIncome'],x['totalRevenue']), axis=1)

    # calculate % growth

    revenue_growth = ['NaN']
    for i in range(1, len(data)):
        revenue_growth.append(per_growth(data['totalRevenue'][i],data['totalRevenue'][i-1]))
    data['revenue_growth'] = revenue_growth

    grossProfit_growth = ['NaN']
    for i in range(1, len(data)):
        grossProfit_growth.append(per_growth(data['grossProfit'][i],data['grossProfit'][i-1]))
    data['grossProfit_growth'] = grossProfit_growth


    operatingIncome_growth = ['NaN']
    for i in range(1, len(data)):
        operatingIncome_growth.append(per_growth(data['operatingIncome'][i],data['operatingIncome'][i-1]))
    data['operatingIncome_growth'] = operatingIncome_growth


    netIncome_growth = ['NaN']
    for i in range(1, len(data)):
        netIncome_growth.append(per_growth(data['netIncome'][i],data['netIncome'][i-1]))
    data['netIncome_growth'] = netIncome_growth

    return data    





tickers_df = get_tickets(INDICES_NAMES)
#holidays_df = get_holidays(tickers_df,ticker_index)

# Layout
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls2 = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Ticker"),
                dcc.Dropdown(
                    id="ticker-variable",
                    options=[
                        {"label": col, "value": col} for col in tickers_df["name"]
                    ],
                    value="ticker",
                ),
            ]
        ),
    ],
    body=True,
)


app.layout = dbc.Container(
    [
        html.H1("Stock forcasting"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls2, md=4),
                dbc.Col(dcc.Graph(id="sector-ticker-graph"), md=8),
                dbc.Col(dcc.Graph(id="forcast-ticker-graph"), md=12),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


# same sector ticker graph
@app.callback(
    Output("sector-ticker-graph", "figure"),
    [
        Input("ticker-variable", "value"),
    ],
)
def make_sector_graph(ticker_name):
    # minimal input validation, make sure there's at least one cluster
    ticker_index = tickers_df[tickers_df.name == ticker_name].index[0]
    #sector_filter = tickers_df[tickers_df.name == ticker_name].sector[0]
    tickers_data = get_ticker_regressors(tickers_df, ticker_index, FILTER)
    #print(f'{len(tickers_data)} regressor tickers added')

    data = [
        go.Scatter(
            x=tickers_data[ticker_index]['data'].index,
            y=tickers_data[ticker_index]['data'].close,
            mode="markers",
            marker={"size": 5},
            name=f"{ticker_name} - {ticker_index}",
        )
    ]
    for ticker_reg_index in tickers_data.keys():
        data.append(
            go.Scatter(
                x=tickers_data[ticker_reg_index]['data'].index,
                y=tickers_data[ticker_reg_index]['data'].close,
                mode="lines",
                line=go.scatter.Line(color='gray'),
                name=f"Ticker {ticker_reg_index}",
            )
        )
    #print(data)
    layout = {"xaxis": {"title": "date"}, "yaxis": {"title": "value"}}

    return go.Figure(data=data, layout=layout)

# same sector ticker graph
@app.callback(
    Output("forcast-ticker-graph", "figure"),
    [
        Input("ticker-variable", "value"),
    ],
)
def make_forcast_graph(ticker_name):
    # minimal input validation, make sure there's at least one cluster
    ticker_index = tickers_df[tickers_df.name == ticker_name].index[0]
    #sector_filter = tickers_df[tickers_df.name == ticker_name].sector[0]
    tickers_data = get_ticker_regressors(tickers_df, ticker_index, FILTER)
    #print(f'{len(tickers_data)} regressor tickers added')
    holidays_df = get_holidays(tickers_df,ticker_index)
    
    p_df = pd.DataFrame({
        "ds":tickers_data[ticker_index]['data'].index,
        "y": tickers_data[ticker_index]['data'].close
    }).reset_index(drop=True)

    fin_regressors = get_fin_regressors(ticker_index)
    print(fin_regressors)

    # add grossProfit_margin to the df
    
    p_df['grossProfit_margin'] = 'NaN'
    for d in fin_regressors['endDate']:
        mask = p_df['ds'] > d
        p_df['grossProfit_margin'][mask] = fin_regressors.loc[d.strftime("%Y-%m-%d")].grossProfit_margin

    print(p_df)

    data = [
        go.Scatter(
            x=p_df["ds"],
            y=p_df["y"],
            mode="markers",
            marker={"size": 5},
            name=f"{ticker_name} - {ticker_index}",
        )
        #for c in range(n_clusters)
    ]
    # for ticker_reg_index in tickers_data.keys():
    #     data.append(
    #         go.Scatter(
    #             x=tickers_data[ticker_reg_index]['data'].index,
    #             y=tickers_data[ticker_reg_index]['data'].close,
    #             mode="lines",
    #             line=go.scatter.Line(color='gray'),
    #             name=f"Ticker {ticker_reg_index}",
    #         )
    #     )
    # print(data)
    layout = {"xaxis": {"title": "date"}, "yaxis": {"title": "value"}}

    return go.Figure(data=data, layout=layout)



if __name__ == "__main__":
    app.run_server(debug=True, port=8888)