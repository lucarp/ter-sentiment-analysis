import os
import re
from numpy import genfromtxt
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from collections import Counter
# nltk.download('punkt')
# nltk.download('stopwords')



def countWordsOnReviews(df):
    wordCounter = Counter()
    for review in df['Review']:
        for word in word_tokenize(review):
            wordCounter[word] +=1
    
    return wordCounter

def cleanseData(df, threshold):
    counter = countWordsOnReviews(df)
    vocab = {x : counter[x] for x in counter if counter[x] >= threshold }
    print('Vocabulary size: ' + str(len(vocab)))
    f = open( 'vocab.json', 'w' )
    f.write(repr(vocab))
    f.close()
    i = 0
    new_df_review = []
    for review in df['Review']:
        new_review = []
        for word in word_tokenize(review):
            if word in vocab:
                new_review.append(word)
        review = ' '.join(new_review)
        new_df_review.append(review)
    new_df_review = pd.DataFrame(new_df_review, columns=['Review'])
    df.drop(['Review'], 1, inplace=True)
    df.reset_index(inplace=True, drop=True)   
    df = pd.concat([df, new_df_review], axis=1)
    return df

def importDataset(path_to_dataset):
    path_to_reviews = path_to_dataset + '/scale_whole_review'
    path_to_scaledata = path_to_dataset + '/scaledata'
    authors = os.listdir(path_to_scaledata)

    frames = []
    for author in authors:
        # Skip hidden files
        if author.startswith('.'):
            continue

        ids = pd.read_csv(path_to_scaledata  + '/' + author + '/id.' + author, header=None)
        ratings = pd.read_csv(path_to_scaledata + '/' + author + '/rating.' + author, header=None)
        reviews = []
        for reviewId in ids[0]:
            print('Extracting review ' + str(reviewId) + ' by ' + author)
            review_file = open(path_to_reviews  + '/' + author + '/txt.parag/' + str(reviewId) + '.txt', encoding='latin-1')
            reviews.append(preprocess(word_tokenize(review_file.read())))
            review_file.close()
        frames.append(pd.DataFrame(data={'ID': ids[0], 'Author': author, 'Rating': ratings[0], 'Review': reviews}))
    df = pd.concat(frames)
    df = cleanseData(df,30)
    df.to_csv('output.csv')
    
def preprocess(words):
    new_words = []
    for word in words:
        
        # Remove from array punctuation words
        temp = re.sub(r'[^\w\s]', '', word)
        if temp == '':
            continue

        # To lowercase
        temp = word.lower()

        # Remove line breaks
        temp = temp.replace('\n', ' ').replace('\r', '').replace('\t', ' ')

        # Remove numbers
        if temp.isdigit():
            continue

        # Remove stop words
        if temp in stopwords.words('english'):
            continue

        new_words.append(temp)
        # Retunr a single string with preprocessed text
    return ' '.join(str(x) for x in new_words)

importDataset('dataset')
