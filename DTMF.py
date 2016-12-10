#!/usr/bin/bash
#Modified by MonsieurVechai (taken from https://github.com/boxcarton/speech-sentiment-python/blob/master/speech_sentiment_python/recorder.py)
from sys import byteorder
from array import array
from struct import pack
import os
import pyaudio
import wave
import math
import struct

#Uncomment the next line if used on Raspberry Pi
#os.system("modprobe snd_bcm2835")
THRESHOLD = 400
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 4096

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in xrange(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            print "SOUND DETECTED"
            snd_started = True
        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


class pygoertzel_dtmf:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.goertzel_freq = [1209.0,1336.0,1477.0,1633.0,697.0,770.0,852.0,941.0]
        self.s_prev = {}
        self.s_prev2 = {}
        self.totalpower = {}
        self.N = {}
        self.coeff = {}
        # create goertzel parameters for each frequency so that
        # all the frequencies are analyzed in parallel
        for k in self.goertzel_freq:
            self.s_prev[k] = 0.0
            self.s_prev2[k] = 0.0
            self.totalpower[k] = 0.0
            self.N[k] = 0.0
            normalizedfreq = k / self.samplerate
            self.coeff[k] = 2.0*math.cos(2.0 * math.pi * normalizedfreq)
    def __get_number(self, freqs):
        hi = [1209.0,1336.0,1477.0,1633.0]
        lo = [697.0,770.0,852.0,941.0]
        # get hi freq
        hifreq = 0.0
        hifreq_v = 0.0
        for f in hi:
            if freqs[f]>hifreq_v:
                hifreq_v = freqs[f]
                hifreq = f
        # get lo freq
        lofreq = 0.0
        lofreq_v = 0.0
        for f in lo:
            if freqs[f]>lofreq_v:
                lofreq_v = freqs[f]
                lofreq = f
        if lofreq==697.0:
            if hifreq==1209.0:
                return "1"
            elif hifreq==1336.0:
                return "2"
            elif hifreq==1477.0:
                return "3"
            elif hifreq==1633.0:
                return "A"
        elif lofreq==770.0:
            if hifreq==1209.0:
                return "4"
            elif hifreq==1336.0:
                return "5"
            elif hifreq==1477.0:
                return "6"
            elif hifreq==1633.0:
                return "B"
        elif lofreq==852.0:
            if hifreq==1209.0:
                return "7"
            elif hifreq==1336.0:
                return "8"
            elif hifreq==1477.0:
                return "9"
            elif hifreq==1633.0:
                return "C"
        elif lofreq==941.0:
            if hifreq==1209.0:
                return "*"
            elif hifreq==1336.0:
                return "0"
            elif hifreq==1477.0:
                return "#"
            elif hifreq==1633.0:
                return "D"
    def run(self, sample):
        freqs = {}
        for freq in self.goertzel_freq:
            s = sample + (self.coeff[freq] * self.s_prev[freq]) - self.s_prev2[freq]
            self.s_prev2[freq] = self.s_prev[freq]
            self.s_prev[freq] = s
            self.N[freq]+=1
            power = (self.s_prev2[freq]*self.s_prev2[freq]) + (self.s_prev[freq]*self.s_prev[freq]) - (self.coeff[freq]*self.s_prev[freq]*self.s_prev2[freq])
            self.totalpower[freq]+=sample*sample
            if (self.totalpower[freq] == 0):
                self.totalpower[freq] = 1
            freqs[freq] = power / self.totalpower[freq] / self.N[freq]
        return self.__get_number(freqs)

if __name__ == '__main__':
    print("please speak a word into the microphone")
    record_to_file('demo.wav')
    print("done - result written to demo.wav")
    print("Analyzing")
    
    wav = wave.open('demo.wav', 'r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    frames = wav.readframes(nframes * nchannels)
    # convert wave file to array of integers
    frames = struct.unpack_from("%dH" % nframes * nchannels, frames)
    # if stereo get left/right
    if nchannels == 2:
        left = [frames[i] for i in range(0,len(frames),2)]
        right = [frames[i] for i in range(1,len(frames),2)]
    else:
        left = frames
        right = left
    binsize = 400
    # Split the bin in 4 to average out errors due to noise
    binsize_split = 4
    prevvalue = ""
    prevcounter = 0
    for i in range(0,len(left)-binsize,binsize/binsize_split):
        goertzel = pygoertzel_dtmf(framerate)
        for j in left[i:i+binsize]:
            value = goertzel.run(j)
        if value==prevvalue:
            prevcounter+=1
            if prevcounter==10:
                print value
        else:
            prevcounter=0
            prevvalue=value
