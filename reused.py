# code here...

import yfinance as yf
import pandas as pd
import numpy as np

class MustHaveTwoIndices(Exception):
    '''
    Exception raised when covariance is given a dataframe that does not only 
    contain 2 indices.
    '''

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InvalidDataNaN(Exception):
    '''
    Exception raised when loading a dataset from yfinance library and the 
    dataset includes NaN values
    '''

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

        def __str__(self):
            return self.message

class InvalidDataZN(Exception):
    '''
    Exception raised when loading a dataset from the yfinance library and the 
    dataset includes zero or negative values.
    '''
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

def typical_price(price: pd.Series) -> list:
    '''
    Calculates typical price of an index on a certain day. 

    Parameters: 
    price: A DataFrame object with ticker price data

    Returns: 
    Series of typical prices for the tickers.
    '''
    
    tickers: list(str) = price.index.get_level_values(1).unique().tolist()
    
    typ_prices: list(float) = []

    for t in tickers:
        tp = (price["Close", t] + price["High", t] + price["Low", t]) / 3
        typ_prices.append(tp)

    return typ_prices

def load_stock_data(ticker: list[str], start_date:str, end_date:str) -> pd.DataFrame:
    """
    Load stock data for a given ticker and date range.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL').
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    pd.Series: a series, containing typical price of the stock over time.
    Only returns pd.DataFrame when loading price data for multiple indices 
    at once.
    """

    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.isna().any().any():
        raise InvalidDataNaN(
                "This dataset is incomplete."
                "There are NaN values in this dataset.")
    if (stock_data <= 0).any().any():
        raise InvalidDataZN(
                "This dataset has errors. There are"
                "Zero and Negative values.")
    
    tickers: list = stock_data.columns.get_level_values(1).unique().tolist()

    for t in tickers:
        stock_data["typ", t] = stock_data.apply(lambda row: typical_price(row)[tickers.index(t)], axis = 1)

    return stock_data["typ"]


def log_returns(mkt_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates log returns for a market dataframe to normalize changes. 

    Parameters:
    mkt_data: a dataframe containing typical price for a ticker, with 
    information from the yfinance library. 

    Returns:
    A dataframe indexed by date and containing daily log returns
    '''

    columns: list[str] = mkt_data.columns.values.tolist()

    mkt_ret: pd.DataFrame = pd.DataFrame(data=[None] * mkt_data.shape[0], 
                                         index = mkt_data.index, 
                                         columns = columns)
    
    for i in range(1, mkt_data.shape[0]):
        for column in columns:
            ticker: pd.DataFrame = mkt_data[column]
            tick_ret = float(np.log(ticker.iloc[i]/ticker.iloc[i-1]))
            mkt_ret.iloc[i, mkt_ret.columns.get_loc(column)] = tick_ret

    return mkt_ret

def covariance(mkt_data: pd.Series, time_window: int) -> pd.Series: 
    '''
    Calculates the rolling covariance of two indices, within a certain time 
    window. The function is input agnostic (can process typical price, log 
    returns, and simple returns). 

    Parameters:
    market_data: a dataframe containing market information from the yfinance
    library for both markets / indices being measured.
    time_window: the rolling time window for covariance measure (will be optimized).

    Returns:
    A series of covariance over time. 
    '''

    # Measuring covariance only between two indices.
    tickers = mkt_data.columns.values.tolist()
    if len(tickers) != 2:
        raise MustHaveTwoIndices("Covariance must only measure two indices")
    
    covariance: None | list = []
    # Computing the rolling time window for covariance. 
    for i in range(prices.shape[0]): # pct_change drops first row.
        if i < time_window:
            covariance.append(None)
            continue
        
        window = prices.iloc[i - time_window: i]
        means = window.mean()
        t1 = tickers[0]
        t2 = tickers[1]
        
        # covariance computation
        cov = ((window[t1] - means[t1]) * (window[t2] - means[t2])).sum() / (time_window - 1)
        covariance.append(float(cov))
    
    return pd.DataFrame(covariance, index = prices.index, name = "Covariance").dropna()

def variance(mkt: pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    '''
    Calculates rolling variance of an index according to a rolling time window. 

    Parameters:
    mkt: a dataframe of price data for the market

    Returns:
    A series or dataframe, showing log-normalized variance over time
    '''
    if isinstance(mkt, pd.Series):
        df = mkt.to_frame()
    else:
        df = mkt

    tickers = df.columns.values.tolist()

    variance: pd.DataFrame = pd.DataFrame(index = mkt.index, columns = tickers)
    
    for t in tickers:
        var_list_t: list = []
        for i in range(mkt.shape[0]):
            if i < window:
                var_list_t.append(None)
                continue
        
            period = mkt[t].iloc[i - window: i]
            mean_t = period.mean()

            var = float(((period - mean_t) ** 2).sum()) / (window - 1)

            var_list_t.append(var)


        variance.isetitem(tickers.index(t), var_list_t)
    
    return variance.dropna()

def beta(market_data: pd.DataFrame, time_window: int, ticker: str) -> pd.Series:
    '''
    given price data for the market and a specific asset, return
    a list of beta over time. 

    Parameters:
    market_data: dataframe from yfinance of market index, and asset index
    ticker: a string, denoting which ticker is supposed to be the market.

    Returns:
    Series of beta measures over time.
    '''
    # beta = cov(Ri, Rm) / var(Rm)

    try:
        cov = covariance(market_data, time_window)
    except MustHaveTwoIndices:
        print("not a valid dataframe for covariance and volatility measure!")
        return None

    var = variance(market_data, time_window)
    
    beta_series = pd.Series(cov / var[ticker], index = cov.index, name = "beta")

    return beta_series

def volatility(market_data: pd.DataFrame, time_window: int) -> pd.Series | pd.DataFrame:
    '''
    Calculates rolling volatility of an asset or index. This is simply the standard 
    deviation calculation.

    Parameters:
    market_data: dataframe containing typical price data for some asset or index

    Returns:
    Series or Dataframe with volatility of the asset.
    '''
    var = variance(market_data, time_window)

    # standard deviation -> volatility is the same as the square root of
    # variance. 
    return var.pow(0.5)
