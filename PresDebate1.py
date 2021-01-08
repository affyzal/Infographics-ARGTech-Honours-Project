import csv
import numpy as np
import pandas as pd
import os
import spacy
import matplotlib
import nltk
import string
import plotly
import warnings
import datetime
import wordcloud


def main():
    debate1 = readfile("us_election_2020_1st_presidential_debate.csv")
    dfcw, dfdt, dfjb = isolatespeaker(debate1)
    dftext = isolatetext(dfcw)

def readfile(filetoread):
    file = filetoread
    df = pd.read_csv(file)
    return df


#TODO : Add logic to work for every debate
def isolatespeaker(debatefile):
    df = debatefile
    dfcw = df[df['speaker'] == 'Chris Wallace']
    dfdt = df[df['speaker'] == 'President Donald J. Trump']
    dfjb = df[df['speaker'] == 'Vice President Joe Biden']
    return dfcw, dfdt, dfjb


def fixnull():
    # find null values in csv
    # fix
    return


def tidy():
    # tidy up unnecessary words
    return


def isolatetext(speakerfile):
    df = speakerfile
    df2 = df['text']
    return df2


if __name__ == '__main__':
    main()
