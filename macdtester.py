from binance.client import Client
from downloader import Downloader
import matplotlib.pyplot as plt
dl = Downloader()
import pandas as pd
import numpy as np
import math
import seaborn as sns
import metalabeller as ml
import ta

## MACD test file
def get_base_macd_events(df, mlen=3, stdev=1):
    """
    returns all macd events satisfying criteria:
    - length > mlen
    - gain is at least stdev standard deviations
    """
    df["macd"] = ta.macd(df.close, result="hist")
    df["macd_past"] = df.macd.shift(1)
    df["ema"] = df.close.ewm(span=500).mean()
    df["ema_diff"] = df.ema.diff()
    df.dropna(inplace=True)
    t0 = df[(df.macd<0) & (df.macd_past>0)].index
    tn1 = df[(df.macd>0) & (df.macd_past<0)].index
    
    stats = ["tn1", "t0", "gain", "length"]
    result = pd.DataFrame(np.nan, index=range(len(t0)), columns=stats)
    ls = tn1.searchsorted(t0, side='left', sorter=None)-1   # location of start (corresponding
                                                            # tn1 for t0
    for i in range(len(t0)):
        if ls[i]<0: continue 
        if tn1[ls[i]]-pd.Timedelta(days=7) not in df.index: continue
        df_ = df.loc[tn1[ls[i]]:t0[i]]
        if df_.shape[0]<mlen: continue
        
        row = dict.fromkeys(stats, 0.)
        row["tn1"] = tn1[ls[i]]
        row["t0"] = t0[i]
        row["gain"] = (max(df_.high)-df_.iloc[0].close)/df_.iloc[0].close
        row["length"] = len(df_)
        result.iloc[i] = row
    
    min_gain = np.mean(result.gain)+stdev*np.std(result.gain)
    result = result[result.gain > min_gain]
    result.dropna(inplace=True)
    result.index = range(len(result))
    return result


def get_macd_events(df, events, stats, funcs, min_length=3, l1=1, l2=3, l3=7):
    """
    stats = ['gain', v0'...]
    funcs = [func_1, func_2...], needs to be same length as stats
    l1, l2, l3 are lengths from tn1-days:t0. 
    All funcs need to have signature func(df_, df_1, df_7, df_30)->float
    
    Returns macd events, dropped nans. 
    """
    if len(stats) != len(funcs): raise Exception
    result = pd.DataFrame(np.nan, index=events.t0, columns=stats)

    for i in range(len(events)):
        if events.tn1[i]-pd.Timedelta(days=30) not in df.index: continue
        df_ = df.loc[events.tn1[i]:events.t0[i]]
        df_1 = df.loc[events.tn1[i]-pd.Timedelta(days=1):events.t0[i]]
        df_2 = df.loc[events.tn1[i]-pd.Timedelta(days=7):events.t0[i]]
        df_3 = df.loc[events.tn1[i]-pd.Timedelta(days=30):events.t0[i]]
        if df_.shape[0]<min_length: continue
        row = dict.fromkeys(stats, 0.)
        
        for j in range(len(stats)):
            row[stats[j]] = funcs[j](df_, df_1, df_7, df_30)

        result.iloc[i] = row
    return result.dropna()

def get_labels(df, events, time_stop, trgt, tp=1, sl=1, stats=True):
    """
    paramaters
    - df (main dataframe of ohlcv)
    - events, with t0 index
    - time_stop (in days)
    - trgt: pandas series for target % gains (0.02 for 2% gain). Default = events.gain*events.cr
    - tp: tp is target*tp
    - sl: sl is target*sl (sl > 0)
    returns labels dataframe with:
    - t0 index
    - ret (return)
    - target
    """
    t1 = ml.get_vertical_barrier(df.close, events.index, time_stop)
    pd.options.mode.chained_assignment = None
    events["t1"] = t1
    events["trgt"] = trgt
    labels = get_bins(events, df, tp, sl)
    if stats:
        print(f"mean: {round(np.mean(labels.ret)*100,4)}%\nmedian: {round(np.median(labels.ret)*100, 4)}%")
        print(labels.target.value_counts())
        fig, ax = plt.subplots()
        ax.set_title('profitability distribution')
        ax.hist(labels.ret, bins=20)
        plt.show()
    return labels
    
def get_bins(events, df, tp, sl):
    """generates labels with t0 index, return, and target"""
    out = pd.DataFrame(index=events.index, columns=["ret", "target"])
    for t in events.index:
        event = events.loc[t]
        df_ = df.loc[t:event.t1]
        tp_price = tp*event.trgt*df_.iloc[0].close+df_.iloc[0].close
        sl_price = -sl*event.trgt*df_.iloc[0].close+df_.iloc[0].close
        tp_ = df_[df_.high>tp_price].index
        sl_ = df_[df_.low<sl_price].index
        if len(tp_)==0 and len(sl_)==0: 
            out.loc[t] = {"ret":(df_.iloc[-1].close-df_.iloc[0].close)/df_.iloc[0].close, "target":0}
        elif len(tp_)==0: 
            out.loc[t] = {"ret":-event.trgt, "target":0}
        elif len(sl_)==0: 
            out.loc[t] = {"ret":event.trgt, "target":1}
        else: 
            out.loc[t] = {"ret":event.trgt if tp_[0]<sl_[0] else -event.trgt, 
                           "target": 0 if tp_[0]<sl_[0] else 0}
    return out