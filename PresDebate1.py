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
    print("Helloworld")
    debate1 = readfile("us_election_2020_1st_presidential_debate.csv")


def readfile(filetoread):
    file = filetoread
    df = pd.read_csv(file)
    print(df)
    return df
    # readfile
    # return readin file

def isolatespeaker(speaker):
    return

def fixnull():
    # find null values in csv
    # fix
    return


def tidy():
    # tidy up unnecessary words
    return


main()
