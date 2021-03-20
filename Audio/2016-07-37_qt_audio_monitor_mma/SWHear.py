"""
this is a stripped down version of the SWHear class.
It's designed to hold only a single audio sample in memory.
check my githib for a more complete version:
    http://github.com/swharden
"""

#import pyaudio
import time
import numpy as np
import threading
#import benchmark


#import time
import ADS1256
import RPi.GPIO as GPIO

# https://stackoverflow.com/questions/25191620/
#   creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order =20
fs = 1000.0       # sample rate, Hz
cutoff = 300  # desired cutoff frequency of the filter, Hz


# import sys
# import time
# from ADS1256_definitions import *
# from pipyadc import ADS1256
# import bench_config as conf

# # Input pin for the potentiometer on the Waveshare Precision ADC board:
# POTI = POS_AIN0|NEG_AINCOM
# # Light dependant resistor of the same board:
# #LDR  = POS_AIN1|NEG_AINCOM
# LDR  = POS_AIN7|NEG_AINCOM
# # Eight channels
# CH_SEQUENCE = (LDR,)
        



def dbfft(x, fs):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """
    win=None
    #ref = 32768
    #ref = 167772150
    ref = 8777215
    #ref = 1
    #fs = 1.0/fs
    N = int(len(x))  # Length of input sequence


#     if win is None:
#         win = np.ones(1, N)
#     if len(x) != len(win):
#             raise ValueError('Signal and window must be of the same length')
#     x = x * win
    
    x=x*np.hamming(len(x))

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
  #  freq = np.arange((N / 2) + 1) / (float(N) / fs)
    freq=np.fft.fftfreq(len(sp),1.0/fs)
    
        
    #freq = np.arange((N / 2)) / (float(N) / fs)
    
    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    
    
    
    s_mag = np.abs(sp)* 2 / np.sum(np.hamming(len(x)))

    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag/ref)

    #return freq, s_dbfs
    
    #print (s_dbfs[500])
    return freq[:int(len(freq)/2)],s_dbfs[:int(len(s_dbfs)/2)]



def getFFT(data,rate):
    


    """Given some data and rate, returns FFTfreq and FFT (half)."""
    #data = data-np.mean(data)
    data=data*np.hamming(len(data))
    fft=np.fft.fft(data)
    fft=np.abs(fft)
    fft=20*np.log10(fft/32768)
    freq=np.fft.fftfreq(len(fft),1.0/rate)
    return freq[:int(len(freq)/2)],fft[:int(len(fft)/2)]

class SWHear():
    """
    The SWHear class is provides access to continuously recorded
    (and mathematically processed) microphone data.
    
    Arguments:
        
        device - the number of the sound card input to use. Leave blank
        to automatically detect one.
        
        rate - sample rate to use. Defaults to something supported.
        
        updatesPerSecond - how fast to record new data. Note that smaller
        numbers allow more data to be accessed and therefore high
        frequencies to be analyzed if using a FFT later
    """

    def __init__(self,device=None,rate=None,updatesPerSecond=10):
        #self.p=pyaudio.PyAudio()
        #self.chunk=100 # gets replaced automatically
        #self.updatesPerSecond=updatesPerSecond
        self.chunksRead=0
        #self.device=device
        self.rate=rate
        self.data = None
        self.fft = None
        
        self.ads = ADS1256.ADS1256()
        self.ads.ADS1256_init()
         ### STEP 1: Initialise ADC object:
        #self.ads = ADS1256(conf)
        ### STEP 2: Gain and offset self-calibration:
        #self.ads.cal_self()
        timestamp = time.strftime("%b-%d-%Y_Time_%H-%M-%S", time.localtime())
        filename = "FFT_" + timestamp + ".csv"
        self.csvfile = open(filename, 'w', 1)
    
        # Get the filter coefficients so we can check its frequency response.
        self.b, self.a = butter_lowpass(cutoff, fs, order)
        self.printheader = 1

    ### SYSTEM TESTS





 

    def close(self):
        """gently detach from things."""
        print(" -- sending stream termination command...")
        self.keepRecording=False #the threads should self-close
        while(self.t.isAlive()): #wait for all threads to close
            time.sleep(.1)
        #self.stream.stop_stream()
        GPIO.cleanup()
        

    ### STREAM HANDLING

    def stream_readchunk(self):
        """reads some audio and re-launches itself"""
        
        
        try:
            if self.data is None and self.fft is None:
                #self.data = benchmark.do_measurement()
                
                # n_loop = 1000
                n_loop =1000
                self.data = []
                timestamp1 = time.time()
                for a in range(1, n_loop):
                    ### STEP 3: Get data
                    #self.data.append(self.ads.read_continue(CH_SEQUENCE)[0])
                    #self.data.append(self.ads.read_sequence( CH_SEQUENCE)[0])
                    data = self.ads.ADS1256_GetAll()
                    self.data.append((data[4]))
                    #self.data.append((self.ads.ADS1256_GetChannalValue(4)))#//+self.ads.ADS1256_GetChannalValue(7)+
                                      #self.ads.ADS1256_GetChannalValue(7)+self.ads.ADS1256_GetChannalValue(7)))
                timestamp2 = time.time()
                print("\n"*9)
                delta = timestamp2 - timestamp1
                print("Delta seconds: {}".format(delta))
                #print(self.data)
                    # Filter the data, and plot both the original and filtered signals.
        #        self.data = butter_lowpass_filter(self.data, cutoff, fs, order)
                #self.data = self.data[100:]
                self.data = self.data-np.mean(self.data)
                self.data = self.data
                   #print(self.data)
                #self.fftx, self.fft = getFFT(self.data,self.rate)
                self.fftx, self.fft = dbfft(self.data,self.rate)
                
                timestamp = time.strftime("%y.%m.%d-%H:%M:%S ", time.localtime())
                if self.printheader == 1:
                    self.csvfile.write(str(timestamp)+''.join([" {0:0.1f}".format(i) for i in self.fftx]) + "\n")
                    self.printheader = 0
                    
                self.csvfile.write(str(timestamp)+''.join([" {0:0.1f}".format(i) for i in self.fft]) + "\n")

        except Exception as E:
            GPIO.cleanup()
            print(" -- exception! terminating...")
            print(E,"\n"*5)
            self.keepRecording=False
        if self.keepRecording:
            self.stream_thread_new()
        else:
            #self.stream.close()
            #self.p.terminate()
            print(" -- stream STOPPED")
        self.chunksRead+=1

    def stream_thread_new(self):
        self.t=threading.Thread(target=self.stream_readchunk)
        self.t.start()

    def stream_start(self):
        """adds data to self.data until termination signal"""
        #self.initiate()
        print(" -- starting stream")
        self.keepRecording=True # set this to False later to terminate stream
        self.data=None # will fill up with threaded recording data
        self.fft=None
        self.dataFiltered=None #same
        self.stream_thread_new()

if __name__=="__main__":
    ear=SWHear(updatesPerSecond=10) # optinoally set sample rate here
    ear.stream_start() #goes forever
    lastRead=ear.chunksRead
    while True:
        while lastRead==ear.chunksRead:
            time.sleep(.01)
        print(ear.chunksRead,len(ear.data))
        lastRead=ear.chunksRead
    print("DONE")
