import csv
import numpy as np
import pandas as pd
import math
import os
import spacy
import matplotlib
import nltk
import string
import plotly
import warnings
import datetime
import wordcloud

#TODO : FIX NAN VALUE IN TIME IN DEBATE AND MAKE DATAFRAME INTO ONE PART
#TODO : FIX TIME VALUES IN DEABATE TO NOT BE SPLIT UP INTO PARTS AND MAKE DATAFRAME INTO ONE PART

def main():
    debate1 = readfile("us_election_2020_1st_presidential_debate.csv")
    dfcw, dfdt, dfjb = isolatespeaker(debate1, 1)
    cwtext = isolatetext(dfcw)
    dttext = isolatetext(dfdt)
    jbtext = isolatetext(dfjb)

    debate2 = readfile("us_election_2020_2nd_presidential_debate.csv")
    dfkw, dfdt2, dfjb2 = isolatespeaker(debate2, 2)
    cwtext2 = isolatetext(dfkw)
    dttext2 = isolatetext(dfdt2)
    jbtext2 = isolatetext(dfjb2)
    print(debate1.speaker.unique())
    print(debate2.speaker.unique())


def readfile(filetoread):
    file = filetoread
    df = pd.read_csv(file)
    return df

#TODO : Add logic to work for every debate
def isolatespeaker(debatefile, debatenum):
    df = debatefile
    if debatenum == 1:
        host = df[df['speaker'] == 'Chris Wallace']
        dfdt = df[df['speaker'] == 'President Donald J. Trump']
        dfjb = df[df['speaker'] == 'Vice President Joe Biden']
    else:
        host = df[df['speaker'] == 'Kristen Welker']
        dfdt = df[df['speaker'] == 'Donald Trump']
        dfjb = df[df['speaker'] == 'Joe Biden']

    #dfdt = df[df['speaker'] == 'President Donald J. Trump']
    #dfjb = df[df['speaker'] == 'Vice President Joe Biden']
    return host, dfdt, dfjb


def fixtimeframe():
    # find null values in csv
    # fix
    return


def tidyjunk():
    # tidy up unnecessary words
    return


def isolatetext(speakerfile):
    df = speakerfile
    df2 = df['text']
    return df2


if __name__ == '__main__':
    main()
