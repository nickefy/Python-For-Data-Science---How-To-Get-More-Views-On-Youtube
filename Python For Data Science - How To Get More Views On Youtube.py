# To manipulate data, NLP and visualizations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from subprocess import check_output
# Standard plotly imports
import plotly as py
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import cufflinks
# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)
# To interactive buttons
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import warnings
warnings.filterwarnings("ignore")
#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv(filepath)

#Looking for Null values and the types of our data
df.info()
df.head()

df[['views']].iplot(kind="histogram", 
                bins=50, theme="white", 
                histnorm='probability', 
                title="Distribuition of Views",
                xTitle='Distribution',
                yTitle='Probability')

df['removed_views_outliers'] = RemoveOutliers(df['views'])
df['removed_views_outliers'].iplot(kind="histogram", 
                        bins=100, theme="white", 
                        histnorm='probability', 
                 title= "Distribution of Views Without Outliers",
                        xTitle='Distribution',
                        yTitle='Probability')

def RemoveOutliers(df_num): 
    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)
    # setting the threshold value
    threshold = data_std * 3
    #setting lower and upper limit
    lower, upper = data_mean - threshold, data_mean + threshold
    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    return outliers_removed

# plotting distribution without outliers
df['removed_views_outliers'] = RemoveOutliers(df['views'])
df['removed_views_outliers'].iplot(kind="histogram", 
                        bins=100, theme="white", 
                        histnorm='probability', 
                 title= "Distribution of Views Without Outliers",
                        xTitle='Distribution',
                        yTitle='Probability')

# pie chart
rename_channels = {1:'Film/Animation', 2:'Cars/Vehicles', 10:'Music', 15:'Pets/Animals',
                   17:'Sport',19:'Travel/Events',20: 'Gaming',22:'People/Blogs',23:'Comedy',
                   24:'Entertainment',25:'News/Politics',26:'How to/Style',27:'Education',
                   28:'Science/Technology',29:'Non Profits/Activism'}
df['category_name'] = np.nan
df['category_name'] = df['category_id'].map(rename_channels)
percent_category = round(df["category_name"].value_counts(), 2)
categories = round(df["category_name"].value_counts() / len(df["category_name"]) * 100,2)
labels = list(categories.index)
values = list(categories.values)
trace1 = go.Pie(labels=labels, 
                values=values, 
                marker=dict(colors=['red']), 
                text=(percent_category.index.map(str)),
                hovertext=(percent_category.values.astype(str)))
layout = go.Layout(title="Views by Category", 
                   legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)

# plotting views and channels by category
trace1 = go.Bar(x=df.groupby(['category_name'])['views'].sum().index,
                y=df.groupby(['category_name'])['views'].sum().values,
                name='Total Views by Category', visible=True)
trace2 = go.Bar(x=df.groupby(['category_name'])['channel_title'].nunique().index,
                y=df.groupby(['category_name'])['channel_title'].nunique().values, 
                name='Unique Channels by Category', visible=False)
data = [trace1, trace2]
updatemenus = list([
    dict(active=-1,
         showactive=True,
         buttons=list([  
            dict(
                label = 'Total Views by Category',
                 method = 'update',
                 args = [{'visible': [True, False, False]}, 
                     {'title': 'Sum of views by Category'}]),
             
             dict(
                  label = 'Total Channels by Category',
                 method = 'update',
                 args = [{'visible': [False, True, False]},
                     {'title': 'Total Channels by Category'}]),
]),
    )
])
layout = dict(title='Different Metrics by each category <br> Sum of views by Category', 
              showlegend=False,
              updatemenus=updatemenus)
fig = dict(data=data, layout=layout)
iplot(fig

# defining saturation Metrics
views_list = np.array(df.groupby(['category_name'])['views'].sum().tolist())
channels_list = np.array(df.groupby(['category_name'])['channel_title'].nunique().tolist())
videos_list = np.array(df.groupby(['category_name'])['title'].nunique().tolist())
views_by_videos = views_list/videos_list
views_by_channels = views_list/channels_list

trace1 = go.Bar(x=df.groupby(['category_name'])['title'].nunique().index,
                y=views_by_videos,
                name='Saturation - Views per Video', visible=True)
trace2 = go.Bar(x=df.groupby(['category_name'])['title'].nunique().index,
                y=views_by_channels,
                name='Saturation - Views per Channel', visible=True)
data = [trace1,trace2]
updatemenus = list([
    dict(active=-1,
         showactive=True,
         buttons=list([  
            dict(
                label = 'Saturation - Views per Video',
                 method = 'update',
                 args = [{'visible': [True, False]}, 
                     {'title': 'Saturation - Views per Video'}]),
             
             dict(
                  label = 'Saturation - Views per Channel',
                 method = 'update',
                 args = [{'visible': [False, True]},
                     {'title': 'Saturation - Views per Channel'}]),
        ]),
    )
])
layout = dict(title='*(Select from Dropdown)* Saturation Metrics by Category', 
              showlegend=False,
              updatemenus=updatemenus)
fig = dict(data=data, layout=layout)
iplot(fig)

# Word Count distribution in Video Titles and Tags
#Unique word count
df['count_unique_word']=df["title"].apply(lambda x: len(set(str(x).split())))
df['count_unique_word_tags']=df["tags"].apply(lambda x: len(set(str(x).split())))
df[['count_unique_word_tags','count_unique_word']].iplot(kind="histogram",bins=50, theme="white", histnorm='probability', title="Distribuitions of Word Count in Title and Tags",
xTitle='Distribuition',
yTitle='Probability')

# Word Count vs Views
# Dataframe for unique word count for video titles
df3 = df[['count_unique_word','views']]
df3 = df3.groupby('count_unique_word').mean().reset_index()
# Dataframe for unique word count for video tags
df4 = df[['count_unique_word_tags','views']]
df4 = df4.groupby('count_unique_word_tags').mean().reset_index()
trace1 = go.Bar(x=df3['count_unique_word'],y=df3['views'],name = 'Views vs Video Title Word Count',visible = True)
trace2 = go.Bar(x=df4['count_unique_word_tags'],y=df4['views'],name = 'Views vs Video Tags Word Count',visible = True)
data=[trace1,trace2]
updatemenus = list([
    dict(active=-1,
         showactive=True,
         buttons=list([  
            dict(
                label = 'Views vs Video Title Word Count',
                 method = 'update',
                 args = [{'visible': [True, False]}, 
                     {'title': 'Views vs Video Title Word Count'}]),
             
             dict(
                  label = 'Views vs Video Tags Word Count',
                 method = 'update',
                 args = [{'visible': [False, True]},
                     {'title': 'Views vs Video Tags Word Count'}]),
        ]),
    )
])
layout = dict(title="*(Select from Dropdown)* Views vs Word Count", 
              showlegend=False,
              updatemenus=updatemenus)
fig = dict(data=data, layout=layout)
iplot(fig)

# Word Cloud for Video Title
plt.rcParams['font.size']= 15              
plt.rcParams['savefig.dpi']= 100         
plt.rcParams['figure.subplot.bottom']= .1 
stopwords = set(stopwords.words("english"))
plt.figure(figsize = (15,15))
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=1000,
                          max_font_size=120, 
                          random_state=42
                         ).generate(str(df['title']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - VIDEO TITLE")
plt.axis('off')
plt.show()

# Word Cloud for Video Tags
plt.figure(figsize = (15,15))
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=1000,
                          max_font_size=120, 
                          random_state=42
                         ).generate(str(df['tags']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TAGS")
plt.axis('off')
plt.show()