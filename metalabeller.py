import pandas as pd
import numpy as np
from multiprocess import mp_pandas_obj


def get_daily_vol(close, span=10000, days=2, hours=0):
    """
    daily vol, reindexed to close

    Arguments:
    close -- daily close (probably Pandas series)
    span -- span lol (probably int)
    Purpose:
    use the output of this function to set default profit taking and stop-loss limit
    """
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=days, hours=hours))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span).std()
    return df0

def get_t_events(close, h, m, neg=True, pos=True):
    """
    Implementation of the symmetric CUSUM filter seen in chapter 2.5.2

    Arguments:
    close -- the raw time series to filter (possibly pandas series)
    h -- threshold pandas series vol
    m -- minimum threshold (float, ex: 0.02)
    neg -- include neg indicators
    pos -- include pos indicators

    Purpose:
    The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of a measured quantity away from a target value. 
    """
    h = h.map(lambda x: x if x>=m else m)
    t_events, s_pos, s_neg = [], 0, 0
    diff = np.log(close).diff().dropna()
    for i in diff.index[1:]:
        try:
            pos, neg = float(s_pos+diff.loc[i]), float(s_neg+diff.loc[i])
        except Exception as e:
            print(e)
            print(s_pos+diff.loc[i], type(s_pos+diff.loc[i]))
            print(s_neg+diff.loc[i], type(s_neg+diff.loc[i]))
            break
        s_pos, s_neg=max(0., pos), min(0., neg)
        try: 
            th = h.loc[i]
        except:
            try:
                th = h[h.index.get_loc(i, method="pad")]
            except: #i is before the first ever h
                continue
        if s_neg<-th:
            s_neg=0
            if neg: t_events.append(i)
#             side.append(0)
        if s_pos>th:
            s_pos=0
            if pos: t_events.append(i)
#             side.append(1)
#     return pd.DatetimeIndex(t_events)
    t_events = remove_duplicates_from_list(t_events) #remove duplicates, crude solution
    return t_events

def remove_duplicates_from_list(seq):
    """remove duplicates from list whilst keeping order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_vertical_barrier(close, t_events, num_days, hours=0):
    """
    Snippet 3.4, indended to be used on t1
    """
    shift = pd.Timedelta(days=num_days, hours=hours)
    t1=close.index.searchsorted(list(map(lambda x: x+shift, t_events)))
    print("verify that get_vertical_barrier works as expected, use above code")
#     t1=close.index.searchsorted(t_events+pd.Timedelta(days=num_days))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=t_events[:t1.shape[0]]))
    t1 = t1[~t1.index.duplicated()] #remove duplicates, crude solution
    print("remove above remove duplicates code with BTC data and see if it works")
    return t1

def get_events(close, t_events, ptsl, trgt, min_ret, num_threads, t1, side):
    """
    finds the time of the first barrier touch
    
    Arguments:
    close -- a pandas series of prices
    t_events -- the pandas timeindex containing the timestamps 
                that will seed every triple barrier. These are the 
                timestamps discussed in section 2.5
    ptsl -- a non-negative float that sets the width of the two 
                barriers. A 0 value means that the respective 
                horizontal barrier (profit taking and/or stop loss) 
                will be disabled
    t1 -- a pandas series with the timestamps of the vertical barriers. 
            We pass a False when we want to disable vertical barriers
    trgt -- a pandas series of targets, expressed in terms of absolute returns
    min_ret -- the minimum target return required for running a triple barrier search
    num_threads -- the number of threads concurrently used by the function
    
    Output:
    pandas dataframe with columns
    t1 -- the timestamp at which the first barrier is touched
    trgt -- the target that was used to generate the horizontal barriers
    """
    #1) get target
    trgt=trgt.reindex(t_events)
    trgt=trgt[trgt>min_ret] # min_ret
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT, index=t_events)
    #3) form events object, apply stop loss on t1
    if side is None:side_,ptsl_=pd.Series(1.,index=trgt.index), [ptsl[0],ptsl[0]]
    else: side_,ptsl_=side.reindex(trgt.index),ptsl[:2] #side.loc[trgt.index],ptsl[:2]
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
            .dropna()) #Note this drops most recent as well
    events["t1"] = events["t1"]
    df0 = mp_pandas_obj(func=apply_triple_barrier,pd_obj=('molecule',events.index),
                    num_threads=num_threads,close=close,events=events,
                    ptsl=ptsl_)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None:events=events.drop('side',axis=1)
    return events

def apply_triple_barrier(close, events, ptsl, molecule):
    """
    apply stop loss /profit taking, if it takes place between t1 (end of event)
    
    Arguments:
    close -- pandas series of prices
    events -- pandas dataframe with columns:
        t1: The timestamp of vertical barrier. 
            When the value is np.nan, there will not be a vertical barrier
        trgt: The unit width of the horizontal barriers
    ptsl -- a list of two non-negative float values:
        ptsl[0] -- the factor that multiplies trgt to set the width of 
                    the upper barrier. If 0, there will not be an upper barrier
        ptsl[1] -- the factor that multiples trgt to set the width of 
                    the lower barrier. If 0, there will not be a lower barrier
    molecule -- A list with the subset of event indices that 
                will be processed by a single thread
    
    Output:
    The output from this function is a pandas dataframe containing 
    the timestamps (if any) at which each barrier was touched.
    """
    events0 = events.loc[molecule]
    out = events0[['t1']].copy(deep=True)
    if ptsl[0]>0:
        pt=ptsl[0]*events0['trgt']
    else:
        pt=pd.Series(index=events.index) #NaNs
    if ptsl[1]>0:
        sl=-ptsl[1]*events0['trgt']
    else:
        sl=pd.Series(index=events.index) #NaNs
    for loc, t1 in events0['t1'].fillna(close.index[-1]).iteritems(): #TODO DEBUG THIS
        df0=close.loc[loc:t1] #path prices
        df0=(df0/close[loc]-1)*events0.at[loc, 'side'] #path returns
        out.loc[loc, 'sl']=df0[df0<sl[loc]].index.min() #earliest stop loss
        out.loc[loc, 'pt']=df0[df0>pt[loc]].index.min() #earliest profit taking
    return out


def get_bins(events, close, t1=None): 
    '''
    Adjust the getBins function (Snippet 3.5) to return a 0 whenever the 
    vertical barrier is the one touched first.
    
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['target']=np.sign(out['ret'])
    
    if 'side' not in events_:
        # only applies when not meta-labeling
        # to update bin to 0 when vertical barrier is touched, we need the original
        # vertical barrier series since the events['t1'] is the time of first 
        # touch of any barrier and not the vertical barrier specifically. 
        # The index of the intersection of the vertical barrier values and the 
        # events['t1'] values indicate which bin labels needs to be turned to 0
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'target'] = 0.
    
    if 'side' in events_:out.loc[out['ret']<=0,'target']=0 # meta-labeling
    return out