import pandas as pd
import numpy as np

def macd(close, fast=3, slow=10, signal=16, result="hist"):
    """
    returns the macd for close. 
    result:
    -hist: returns the standard histogram (macd-signal)
    -macd: returns macd
    -signal: returns signal (two values)
    """
    exp1 = close.rolling(window=fast).mean()
    exp2 = close.rolling(window=slow).mean()
    macd = exp1-exp2
    sig = macd.rolling(window=signal).mean()
    hist = macd-sig
    if result=="signal": 
        return sig
    elif result=="macd":
        return macd
    return hist