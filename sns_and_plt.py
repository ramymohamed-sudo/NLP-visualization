# This file has some data visualization codes for Explonatory data analysis (EDA)

import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

url = 'https://raw.githubusercontent.com/laxmimerit/twitter-disaster-prediction-dataset/master/train.csv'
tweeter = pd.read_csv(url)
tweeter.head()


""" sns.countplot """ 
figure = plt.figure()
figure.figsize = [8,4]
figure.dpi = 80
figure.tight_layout()
axes1 = figure.add_subplot(1,2,1)
axes1.set_title("Disaster or not Disaster")
sns.countplot('target',data=tweeter)
# axes.set_xlabel("")
# axes.plot([x],[y])


""" Matplotlib plt.pie """ 
axes2 = figure.add_subplot(1,2,2)
axes2.set_title("Pie Chart")
axes2.pie(tweeter.target.value_counts(),labels=["0","1"],
#explode = [0.1,0],
autopct = "%1.2f%%"	# 1 for numbers and 1f for decimal point
)


!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall
import preprocess_kgptalkie as kgp
tweet = kgp.get_basic_features(tweeter)

""" sns.distplot """
figure = plt.figure()
figure.figsize = [8,4]
figure.dpi = 80
axes = figure.add_subplot(1,1,1)
axes.set_title("Distribution")
sns.distplot(tweet['char_counts'])	

""" sns.kdeplot """
figure = plt.figure()
figure.figsize = [4,4]
figure.dpi = 80
axes = figure.add_subplot(1,1,1)
axes.set_title("Distribution ...")
sns.kdeplot(tweet['char_counts'],shade=True)

sns.kdeplot(tweet[tweet['target']==1]['char_counts'],shade=True)
sns.kdeplot(tweet[tweet['target']==0]['char_counts'],shade=True)

""" cat.kdeplot """
sns.catplot(y='char_counts',data=tweet,kind='violin',col='target')



