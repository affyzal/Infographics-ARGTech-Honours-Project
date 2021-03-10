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


#TODO : MAKE SENTIMENT CODE EFFICIENT WITH FUNCTIONS
#TODO : BIGRAMS
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

    #DoHeatMaps(debate1, debate2)

    SentenceTokenizer(debate1)
    SentenceTokenizer(debate2)

 #  SentenceTokenizer(dfdt)
 #  SentenceTokenizer(dfjb)

 #  SentenceTokenizer(dfdt2)
 #  SentenceTokenizer(dfjb2)

    # summing up the number of sentences
    sentencenum = debate1.groupby(['speaker']).sum()[['sentences']].reset_index()
    sentencenum2 = debate2.groupby(['speaker']).sum()[['sentences']].reset_index()

    print(sentencenum)
    print('#######################')
    print(sentencenum2)
    print('#######################')

    totalsents = sentencenum.sum().sentences
    totalsents2 = sentencenum2.sum().sentences

    print(totalsents)
    print(totalsents2)

    bidencount = sentencenum.at[2, 'sentences'] + sentencenum2.at[1, 'sentences']
    trumpcount = sentencenum.at[1, 'sentences'] + sentencenum2.at[0, 'sentences']

    trumppercentage1 = int(sentencenum.at[1, 'sentences'] / totalsents * 100)
    bidenpercentage1 = int(sentencenum.at[2, 'sentences'] / totalsents * 100)
    mediatorpercentage1 = int(sentencenum.at[0, 'sentences'] / totalsents * 100)

    trumppercentage2 = int(sentencenum2.at[0, 'sentences'] / totalsents * 100)
    bidenpercentage2 = int(sentencenum2.at[1, 'sentences'] / totalsents * 100)
    mediatorpercentage2 = int(sentencenum.at[2, 'sentences'] / totalsents * 100)

    print(trumppercentage2)
    print(bidenpercentage2)
    print(mediatorpercentage2)


    fig = go.Figure(
        data=[go.Bar(x=['First Debate', 'Second Debate'], y=[totalsents, totalsents2])],
        layout=go.Layout(
            title=go.layout.Title(text="# of sentences in total")
        )
    )

    fig.show()


    fig = make_subplots(rows=2, cols=3,
                        specs=[[{"rowspan": 2}, {}, {}],
                        [None, {}, {}]],
                        subplot_titles=("# of sentences in total", "Donald Trump", "Joe Biden", 'First Debate', 'Second Debate'))
    
    fig.add_trace(go.Bar(x=['First Debate', 'Second Debate'],
                     y=[totalsents, totalsents2],
                     text =[totalsents, totalsents2]),
                     row=1, col=1)

    #Donald Trump sentence counts for both debates
    fig.add_trace(go.Bar(x=['First Debate', 'Second Debate', 'Total'],
                     y=[sentencenum.at[1, 'sentences'], sentencenum2.at[0, 'sentences'], trumpcount],
                     text =[sentencenum.at[1, 'sentences'], sentencenum2.at[0, 'sentences'], trumpcount]),
                     row=1, col=2)

    #Joe Biden sentence counts for both debates
    fig.add_trace(go.Bar(x=['First Debate', 'Second Debate', 'Total'],
                     y=[sentencenum.at[2, 'sentences'], sentencenum2.at[1, 'sentences'], bidencount],
                     text =[sentencenum.at[2, 'sentences'], sentencenum2.at[1, 'sentences'], bidencount]),
                     row=1, col=3)

    #Debate 1 sentence % for each speaker
    fig.add_trace(go.Bar(x=['Donald Trump', 'Joe Biden', 'Mediator'],
                     y=[sentencenum.at[1, 'sentences'], sentencenum.at[2, 'sentences'], sentencenum.at[0, 'sentences']],
                     text =[str(trumppercentage1)+'%', str(bidenpercentage1)+'%', str(mediatorpercentage1)+'%']),
                     row=2, col=2)

    #Debate 2 sentence % for each speaker
    fig.add_trace(go.Bar(x=['Donald Trump', 'Joe Biden', 'Mediator'],
                     y=[sentencenum2.at[0, 'sentences'], sentencenum2.at[1, 'sentences'], sentencenum.at[2, 'sentences']],
                     text =[str(trumppercentage2)+'%', str(bidenpercentage2)+'%', str(mediatorpercentage2)+'%']),
                     row=2, col=3)

    imgname = 'SentenceAnalysis.png'
    fig.write_image('img/' + imgname)
    fig.update_traces(textposition='outside', textfont_size=8)
    fig.update_layout(showlegend=False, title_text="Sentence Analysis")
    fig.update_yaxes(title_text='count')
    fig.show()

    fig = make_subplots(rows=2, cols=3,
                        specs=[[{"colspan": 3}, None, None],
                               [{"colspan": 3}, None, None]],
                        subplot_titles=("Presidential Debate 1", "Presidential Debate 2"))

    fig.add_trace(go.Histogram(
            x=debate1[debate1.speaker == 'President Donald J. Trump'].sentences,
            name = 'Trump', xbins=dict(start=-1, end=24, size=1),
            marker_color='#ff1600'),
            row=1, col=1
    )

    fig.add_trace(go.Histogram(
            x=debate1[debate1.speaker == 'Vice President Joe Biden'].sentences,
            name = 'Biden', xbins=dict(start=-1, end=24, size=1),
            marker_color='#2600ff'),
            row=1, col=1,
    )

    fig.add_trace(go.Histogram(
            x=debate1[debate1.speaker == 'Chris Wallace'].sentences,
            name = 'Mediator', xbins=dict(start=-1, end=24, size=1),
            marker_color='#ff00d2'),
            row=1, col=1,
    )

    fig.add_trace(go.Histogram(
            x=debate2[debate2.speaker == 'Donald Trump'].sentences,
            name = 'Trump', xbins=dict(start=-1, end=24, size=1),
            marker_color='#ff1600'),
            row=2, col=1
    )

    fig.add_trace(go.Histogram(
            x=debate2[debate2.speaker == 'Joe Biden'].sentences,
            name = 'Biden', xbins=dict(start=-1, end=24, size=1),
            marker_color='#2600ff'),
            row=2, col=1,
    )

    fig.add_trace(go.Histogram(
            x=debate2[debate2.speaker == 'Kristen Welker'].sentences,
            name = 'Mediator', xbins=dict(start=-1, end=24, size=1),
            marker_color='#ff00d2'),
            row=2, col=1,
    )

    fig.update_layout(title_text="Sentence Analysis")
    fig.update_yaxes(title_text='Sentence Count')
    fig.show()

# sentence = '''The platform provides universal access to the world's best education, partnering with top universities and organizations to offer courses online.'''
  # # Creating a textblob object and assigning the sentiment property
  # analysis = TextBlob(sentence).sentiment
  # print(analysis)+-

def SentenceTokenizer(df):
    dftext = " ".join(content for content in df.text)
    text = dftext.lower()
    sentences = nltk.sent_tokenize(text)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    df['sentences'] = df.text.apply(lambda x: len(sent_detector.tokenize(x)))
    #print(len(sentences))

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
#TODO: Aesthetic Updates
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
        imgname = 'Debate1HeatMap.png'
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
        imgname = 'Debate2HeatMap.png'
    heatmap.write_image('img/' + imgname)
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
