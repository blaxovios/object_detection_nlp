import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
import urllib.request
from matplotlib import pyplot as plt
nltk.download('punkt')
nltk.download("stopwords")
from nltk.corpus import stopwords
import pandas as pd

text_filepath = 'C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/results/deltio_nomikon/deltio 08-03-2022.xlsx'
text_df = pd.read_excel(text_filepath)

labels_df = pd.read_excel('C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/csv/Weekly update 11012022_17022022.xlsx')
labels_list = list(labels_df.columns)
text_df = text_df.reindex(columns=[*text_df.columns.tolist(), 'labels'], fill_value=str(labels_list))
text_description_labels_df = text_df[['Asset Description', 'labels']]
text_description_labels_df = text_description_labels_df[~text_description_labels_df['Asset Description'].duplicated()]

#tokenize text by words
words = word_tokenize(text_description_labels_df['Asset Description'][0])
#check the number of words
print(f"The total number of words in the text is {len(words)}")
#find the frequency of words
fdist = FreqDist(words)
#print the 10 most common words
fdist.most_common(10)
#create an empty list to store words
words_no_punc = []
#iterate through the words list to remove punctuations
for word in words:
    if word.isalpha():
        words_no_punc.append(word.lower())
#print number of words without punctuation
print(f"The total number of words without punctuation is {len(words_no_punc)}")
#find the frequency of words
fdist = FreqDist(words_no_punc)
#Plot the 10 most common words
fdist.plot(10)
plt.show()
#list of stopwords
stopwords_list = stopwords.words("greek")
print(stopwords_list)
#create an empty list to store clean words
clean_words = []
#Iterate through the words_no_punc list and add non stopwords to the new clean_words list
for word in words_no_punc:
    if word not in stopwords_list:
        clean_words.append(word)
print(f"The total number of words without punctuation and stopwords is {len(clean_words)}")
#find the frequency of words
fdist = FreqDist(clean_words)
#Plot the 10 most common words
fdist.plot(10)
plt.show()
#Update the stopwords list
stopwords_list.extend(["said","one","like","came","back"])
#create an empty list to store clean words
clean_words = []
#Iterate through the words_no_punc list and add non stopwords to the new clean_words list
for word in words_no_punc:
    if word not in stopwords_list:
        clean_words.append(word)
#find the frequency of words
fdist = FreqDist(clean_words)
#Plot the 10 most common words
fdist.plot(10)
plt.show()
