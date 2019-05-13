import numpy as np
import pandas as pd
import sys
import re
from nltk.tokenize import word_tokenize 

def texts_to_sentiment(texts_file, sentiment_dict_file, text_column = 0):
	texts_df = pd.read_csv(texts_file, header=0)
	sentiment_df = pd.read_csv(sentiment_dict_file, index_col=0)
	
	doc_sentiment_matrix = []
	dict_term_sentiment_matrix = {}

	# Iterate through all sentiment words in the dictionnary	
	for _, sent_vec in sentiment_df.iterrows():
		sent_word = re.sub("#.*", "", sent_vec[0])
		pos = float(sent_vec[1])
		neg = float(sent_vec[2])
		
		if sent_word in dict_term_sentiment_matrix:
			pos = max(pos, dict_term_sentiment_matrix[sent_word][0])
			neg = max(neg, dict_term_sentiment_matrix[sent_word][1])
		
		neu = 1 if pos == 0 and neg == 0 else 0

		# TODO - save this dict to avoid to compute it every time
		dict_term_sentiment_matrix[sent_word] = (pos, neg, neu)
	
	# Iterate through all the text
	for _, row in texts_df.iterrows():
		temp = [0.,0.,0.]
		text = row[text_column]
		for word in word_tokenize(text):
			if word in dict_term_sentiment_matrix:
				sent_vec = dict_term_sentiment_matrix[word]
				temp = np.add(temp, sent_vec)
		doc_sentiment_matrix.append(temp)	
	
	return doc_sentiment_matrix
		
if __name__ == "__main__":
	text_column = 4
	doc_sent = texts_to_sentiment(sys.argv[1], sys.argv[2], text_column)
	print(doc_sent)
	#pd.DataFrame(doc_sent).to_csv(sys.argv[1]+"doc_sentiment.csv")
	pd.DataFrame(doc_sent).to_csv("doc_sentiment.csv")
