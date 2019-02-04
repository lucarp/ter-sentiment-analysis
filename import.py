import os
import re
from numpy import genfromtxt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')

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
            reviews.append(preprocess(review_file.read()))
            review_file.close()
        
        frames.append(pd.DataFrame(data={'ID': ids[0], 'Author': author, 'Rating': ratings[0], 'Review': reviews}))
    pd.concat(frames).to_csv('output.csv')
    
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

        #TODO: Should we tokenize numbers? Transform numbers into words? Replace by a single token [*NUMBER*] ?


        # Remove stop words
        if word in stopwords.words('english'):
            continue

        new_words.append(temp)
        # Retunr a single string with preprocessed text
    return ''.join(str(x) for x in new_words)



importDataset('dataset')
#    ID  author  rating  doc