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
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textblob
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
from wordcloud import WordCloud


#TODO : FIX NAN VALUE IN TIME IN DEBATE AND MAKE DATAFRAME INTO ONE PART
#TODO : FIX TIME VALUES IN DEABATE TO NOT BE SPLIT UP INTO PARTS AND MAKE DATAFRAME INTO ONE PART
#TODO : MAKE SENTIMENT CODE EFFICIENT WITH FUNCTIONS
#TODO : BIGRAMS
#TODO ; FIX HEATMAP
#TODO : SENTENCE ANALYSIS
#TODO : VISUALISATIONS FOR SENTIMENT, BIGRAMS, HEATMAP, SENTENCE ANALYSIS
#TODO : CLEANUP CODE
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

    #DoWClouds(dfdt, dfdt2, dfjb, dfjb2)

    #DoWCounts(dfdt, dfdt2, dfjb, dfjb2)


    print("Debate 1 - Donald Trump Polarity")
    dftext = " ".join(content for content in dfdt.text)
    sia = SentimentIntensityAnalyzer()
    sent = sia.polarity_scores(dftext)
    sent_val = sent['compound']
    sent.pop('compound')
    print('Sentiment - Polarity = ', sent_val)
    print('Sentiment Split = ' , sent)
    print('####################################')

    print("Debate 2 - Donald Trump Polarity")
    dftext = " ".join(content for content in dfdt2.text)
    sia = SentimentIntensityAnalyzer()
    sent = sia.polarity_scores(dftext)
    sent_val = sent['compound']
    sent.pop('compound')
    print('Sentiment - Polarity = ', sent_val)
    print('Sentiment Split = ' , sent)
    print('####################################')


    print("Debate 2 - Joe Biden Polarity")
    dftext = " ".join(content for content in dfjb.text)
    sia = SentimentIntensityAnalyzer()
    sent = sia.polarity_scores(dftext)
    sent_val = sent['compound']
    sent.pop('compound')
    print('Sentiment - Polarity = ', sent_val)
    print('Sentiment Split = ' , sent)
    print('####################################')


    print("Debate 2 - Joe Biden Polarity")
    dftext = " ".join(content for content in dfjb2.text)
    sia = SentimentIntensityAnalyzer()
    sent = sia.polarity_scores(dftext)
    sent_val = sent['compound']
    sent.pop('compound')
    print('Sentiment - Polarity = ', sent_val)
    print('Sentiment Split = ', sent)
    print('####################################')

    print("Debate 1 - Donald Trump Subjectivity")
    dftext = " ".join(content for content in dfdt.text)
    blob_object = TextBlob(dftext)
    sentences = blob_object.sentences
    analysis = TextBlob(dftext).subjectivity
    print('Subjectivity = ', analysis)
    print('####################################')

    print("Debate 2 - Donald Trump Subjectivity")
    dftext = " ".join(content for content in dfdt2.text)
    blob_object = TextBlob(dftext)
    sentences = blob_object.sentences
    analysis = TextBlob(dftext).subjectivity
    print('Subjectivity = ', analysis)
    print('####################################')

    print("Debate 1 - Joe Biden Subjectivity")
    dftext = " ".join(content for content in dfjb.text)
    blob_object = TextBlob(dftext)
    sentences = blob_object.sentences
    analysis = TextBlob(dftext).subjectivity
    print('Subjectivity = ', analysis)
    print('####################################')

    print("Debate 2 - Joe Biden Subjectivity")
    dftext = " ".join(content for content in dfjb2.text)
    blob_object = TextBlob(dftext)
    sentences = blob_object.sentences
    analysis = TextBlob(dftext).subjectivity
    print('Subjectivity = ', analysis)
    print('####################################')

    #Locate null value and fix
    debate1.loc[debate1.minute.isnull(), 'minute'] = '00:00'

    FixTimeframe(debate1)
    FixTimeframe(debate2)

    DoHeatMaps(debate1, debate2)

  # sentence = '''The platform provides universal access to the world's best education, partnering with top universities and organizations to offer courses online.'''
  # # Creating a textblob object and assigning the sentiment property
  # analysis = TextBlob(sentence).sentiment
  # print(analysis)


def FixTimeframe(df):
    df['seconds'] = 0
    for i, tm in enumerate(df.minute[1:], 1):
        timeParts = [int(s) for s in str(tm).split(':')]

        # when we have hour like 01:10:50
        if (len(timeParts) > 2) and (i < len(df)):

            current = (timeParts[0] * 60 + timeParts[1]) * 60 + timeParts[2]
            difference = current - df.loc[i - 1, 'seconds']
            df.loc[i, 'seconds'] = df.loc[i - 1, 'seconds'] + difference
        # when we get to the second half of the debate
        elif str(tm) == '00:00':
            df.loc[i, 'seconds'] = 0
            second_round_idx = i
            second_round_final_time = df.loc[i - 1, 'seconds']

        # when there's only minute and seconds like 10:50
        elif (i < len(df)):
            current = timeParts[0] * 60 + timeParts[1]
            difference = current - df.loc[i - 1, 'seconds']
            df.loc[i, 'seconds'] = df.loc[i - 1, 'seconds'] + difference

    df.loc[second_round_idx:, 'seconds'] += second_round_final_time
    df['minutes'] = df.seconds.apply(lambda x: x // 60)


#TODO : Potentially add more stop words to WordCloud, Aesthetic/Design Changes
#TODO : TREEMAP VIS
def WCount(df, imgname):
    fig = plt.figure(figsize=(15,10))
    dftext = " ".join(content for content in df.text)
    text = dftext.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if len(word) > 1]
    # count word frequencies
    word_freqs = nltk.FreqDist(words)
    # plot word frequencies
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.title(imgname)
    word_freqs.plot(50)
    plt.gcf()
    fig.savefig('img/' + imgname)
    plt.show()

def DoWCounts(dfdt, dfdt2, dfjb, dfjb2):
    WCount(dfdt, 'Debate 1 - Trump Word Frequency')
    WCount(dfdt2, 'Debate 2 - Trump Word Frequency')
    WCount(dfjb, 'Debate 1 - Biden Word Frequency')
    WCount(dfjb2, 'Debate 2 - Biden Word Frequency')


#TODO : Potentially add more stop words to WordCloud, Aesthetic/Design Changes
def WCloud(content, imgname):
    wordcloud = WordCloud(max_words=100, width=1280, height=720, normalize_plurals=False).generate(content)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud.to_file("img/" + imgname)

def DoWClouds(dfdt, dfdt2, dfjb,dfjb2):
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

def DoHeatMaps(debate1, debate2):
    HeatMap(debate1, 1)
    HeatMap(debate2, 2)

#TODO: Fix Name Prefixes
#TODO: Fix Tick values
def HeatMap(debate, debatenum):
    if debatenum is 1:
        columns = debate.groupby(['minutes', 'speaker']).count().reset_index()
        heatmap = go.Figure(data=go.Heatmap(
            z=columns.minute,
            x=columns.minutes,
            y=columns.speaker,
            colorscale='portland',
            colorbar=dict(
                title="Heatmap of the discussion",
                titleside="top",
                tickmode="array",
                tickvals=[1, 6, 13],
                ticktext=["very cool", "normal", "Hot!"],
                ticks="outside"
            )
        ))
        heatmap.update_layout(title='First Debate: # of times each one talks in each minute', xaxis_nticks=36)
    elif debatenum is 2:
        columns = debate.groupby(['minutes', 'speaker']).count().reset_index()
        heatmap = go.Figure(data=go.Heatmap(
            z=columns.minute,
            x=columns.minutes,
            y=columns.speaker,
            colorscale='portland',
            colorbar=dict(
                title="Heatmap of the discussion",
                titleside="top",
                tickmode="array",
                tickvals=[1, 5, 10],
                ticktext=["very cool", "normal", "Hot!"],
                ticks="outside"
            )
        ))
        heatmap.update_layout(title='Second Debate: # of times each one talks in each minute', xaxis_nticks=36)

    heatmap.show()

def readfile(filetoread):
    file = filetoread
    df = pd.read_csv(file)
    return df

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

#TODO: update to use new method of extraction to isolate as a string and not a data frame
def isolatetext(speakerfile):
    df = speakerfile
    df2 = df['text']
    return df2


if __name__ == '__main__':
    main()
