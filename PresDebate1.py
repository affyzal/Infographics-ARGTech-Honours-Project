import csv

import inline as inline
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
from textblob import TextBlob
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import warnings
import os
#%matplotlib inline

#TODO : FIX NAN VALUE IN TIME IN DEBATE AND MAKE DATAFRAME INTO ONE PART
#TODO : FIX TIME VALUES IN DEABATE TO NOT BE SPLIT UP INTO PARTS AND MAKE DATAFRAME INTO ONE PART
from wordcloud import WordCloud


def main():
    debate1 = readfile("us_election_2020_1st_presidential_debate.csv")
    #FIX TIMEFRAME HERE
    dfcw, dfdt, dfjb = isolatespeaker(debate1, 1)
    cwtext = isolatetext(dfcw)
    dttext = isolatetext(dfdt)
    jbtext = isolatetext(dfjb)

    df = pd.read_csv("winemag-data-130k-v2.csv", index_col = 0)
    df2 = pd.read_csv("us_election_2020_1st_presidential_debate.csv", index_col = 0)

    debate2 = readfile("us_election_2020_2nd_presidential_debate.csv")
    #FIX TIMEFRAME HERE
    dfkw, dfdt2, dfjb2 = isolatespeaker(debate2, 2)
    cwtext2 = isolatetext(dfkw)
    dttext2 = isolatetext(dfdt2)
    jbtext2 = isolatetext(dfjb2)
    
    DoWcs(dfdt, dfdt2, dfjb, dfjb2)


def WCloud(content, imgname):
    wordcloud = WordCloud(max_font_size=30, max_words=100, normalize_plurals=False).generate(content)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud.to_file("img/" + imgname)

def DoWcs(dfdt, dfdt2, dfjb,dfjb2):
    dttext1 = " ".join(content for content in dfdt.text)
    imgname = "TrumpWC1.png"
    WCloud(dttext1, imgname)
    
    dttext2 = " ".join(content for content in dfdt2.text)
    imgname = "TrumpWC2.png"
    WCloud(dttext2, imgname)
    
    jbtext1 = " ".join(content for content in dfjb.text)
    imgname = "BidenWC1.png"
    WCloud(jbtext1, imgname)
    
    jbtext2 = " ".join(content for content in dfjb2.text)
    imgname = "BidenWC2.png"
    WCloud(jbtext2, imgname)
    

def HeatMap(debate):
    heat = debate.groupby(['minute', 'speaker']).count().reset_index()
    heatmap = go.Figure(data=go.Heatmap(
        x=heat.minute,
        #x=heat.minutes,
        y=heat.speaker,
        colorscale='Viridis_r',
        colorbar=dict(
            title="Heat of the discussion",
            titleside="top",
            tickmode="array",
            tickvals=[1, 4, 10],
            ticktext=["very cool", "normal", "Hot!"],
            ticks="outside"
        )
    ))

    heatmap.update_layout(title='First Debate: # of times each one talks in each minute',
                      xaxis_nticks=36)

    heatmap.show()

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
