"""
Created by Bing Liu
Caputre RF data using SDR
"""
from rtlsdr import RtlSdr
import SDRDemodulateFMToAudio as Demo
import SDRHeader as Header
import SDRCommon as com
import time
import warnings

#Collect RF data using SDR
def Capture(sdr, cent_feq, sample_rate, N_Bytes ):
    sdr.center_freq = cent_feq
    sdr.sample_rate = sample_rate
    sdr.gain = 'auto'
    SDR_bytes = sdr.read_bytes(N_Bytes)
    return SDR_bytes
