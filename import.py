import os
import re
import sys
from io import StringIO
from numpy import genfromtxt
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from collections import Counter
# nltk.download('punkt')
# nltk.download('stopwords')



def countWordsOnReviews(df):
    wordCounter = Counter()
    for review in df['Review']:
        for word in word_tokenize(review):
            wordCounter[word] +=1
    
    return wordCounter

def cleanseData(df, threshold, vocab_file):
    counter = countWordsOnReviews(df)
    vocab = {x : counter[x] for x in counter if counter[x] >= threshold }
    print('Vocabulary size: ' + str(len(vocab)))
    f = open(vocab_file, 'w' )
    f.write(repr(vocab))
    f.close()
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

def cleanAndSaveData(fileNameIn, fileNameOut, threshold, vocab_file):
    df = pd.read_csv(fileNameIn, header=0, index_col=0)
    df = cleanseData(df, threshold, vocab_file)
    df.to_csv(fileNameOut)

def importDataset(path_to_dataset, clean_threshold):
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
    df.to_csv('output_without_clean.csv')
    df = cleanseData(df, clean_threshold, 'vocab.json')
    df.to_csv('output.csv')

def importPreProcessedDataset(path_to_dataset, clean_threshold):
    path_to_scaledata = path_to_dataset + '/scaledata'
    authors = os.listdir(path_to_scaledata)

    frames = []
    for author in authors:
        # Skip hidden files
        if author.startswith('.'):
            continue
            
        ids = pd.read_csv(path_to_scaledata  + '/' + author + '/id.' + author, header=None)
        ratings = pd.read_csv(path_to_scaledata + '/' + author + '/rating.' + author, header=None)    
        f = open(path_to_scaledata + '/' + author + '/subj.' + author)
        reviews_file = f.read()
        f.close()
        reviews_file = reviews_file.split('\n')
        reviews_file.remove('') # Remove last line (empty)        
        reviews = []
        i = 0
        for review in reviews_file:
            print('Extracting review ' + str(ids[0][i]) + ' by ' + author)
            i += 1
            reviews.append(preprocess(word_tokenize(review)))
        frames.append(pd.DataFrame(data={'ID': ids[0], 'Author': author, 'Rating': ratings[0], 'Review': reviews}))
    df = pd.concat(frames)
    df.to_csv('output_not_original_without_clean.csv')    
    df = cleanseData(df, clean_threshold, 'vocab_not_original.json')
    df.to_csv('output_not_original.csv')

# Return correct pos for lemmatization
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
    
def preprocess(words):
    new_words = []
    lemmatizer = WordNetLemmatizer()   
    for word in words:
        
        # Remove from array punctuation words
        temp = re.sub(r'[^\w\s]', '', word)
        if temp == '':
            continue

        # To lowercase
        temp = temp.lower()

        # Remove line breaks
        temp = temp.replace('\n', ' ').replace('\r', '').replace('\t', ' ')

        # Remove numbers
        if temp.isdigit():
            continue

        # Remove stop words
        if temp in stopwords.words('english'):
            continue
            
        # Lemmatization
        #temp = lemmatizer.lemmatize(temp, get_wordnet_pos(temp)) # complete lemmatization but slow
        temp = lemmatizer.lemmatize(temp) # fast lemmatization but not perfect

        new_words.append(temp)
        # Return a single string with preprocessed text
    return ' '.join(str(x) for x in new_words)

cleanAndSaveData(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])

"""print("--- Original Dataset")
importDataset('dataset', 30)
print("--- PreProcessed Dataset")
importPreProcessedDataset('dataset', 30)"""
