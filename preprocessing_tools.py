import sys
import pandas as pd
import numpy as np
import scipy
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Process text file and return docs and meta_data
def file_to_data(file_name):
	fichier = open(file_name)
	content = fichier.read()
	fichier.close()
	content = content.split("\n")
	content.remove('') # Remove last line (empty)
	content = content[1:] # Remove header
	content = [i.split(",") for i in content]
	_, author, doc_id, rating, docs = zip(*content)
	meta_data = [list(i) for i in zip(*[rating,author,doc_id])]
	return docs, meta_data
	
# Returns dataFrame from docs and meta_data
def data_to_dataFrame(docs, meta_data, vectorizer):
	X = vectorizer.fit_transform(docs) # non sparse data
	df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
	df_meta = pd.DataFrame(meta_data, columns=['RATING', 'AUTHOR', 'DOC_ID'])
	df = pd.concat([df, df_meta], axis=1)
	return df, X

# Whole pipeline text to dataFrame
def file_to_dataFrame(file_name, vectorizer):
	docs, meta_data = file_to_data(file_name)
	df, X = data_to_dataFrame(docs, meta_data, vectorizer)
	return df, X
	
# Return basic bag of words
def file_to_bow(file_name):
	vectorizer = CountVectorizer()
	df, X = file_to_dataFrame(file_name, vectorizer)
	return df, X
	
# Return tf-idf bag of words
def file_to_tfidf(file_name):
	vectorizer = TfidfVectorizer(norm=None, sublinear_tf=True)
	df, X = file_to_dataFrame(file_name, vectorizer)
	return df, X	
	
# Return tf-idf with l2 norm bag of words	
def file_to_tfidf_l2(file_name):
	vectorizer = TfidfVectorizer(norm='l2', sublinear_tf=True)
	df, X = file_to_dataFrame(file_name, vectorizer)
	return df, X	

def vocab_to_term_sentiment_matrix(vocab_file, sentiment_file):
	vocab_df = pd.read_csv(vocab_file, header=None)
	sentiment_df = pd.read_csv(sentiment_file, index_col=0)
	num_found = 0
	doc_term_sentiment_matrix = []
	
	dict_term_sentiment_matrix = {}

	# Iterate through all sentiment words in the dictionnary	
	for j, sent_vec in sentiment_df.iterrows():
		sent_word = re.sub("#.*", "", sent_vec[0])
		pos = float(sent_vec[1])
		neg = float(sent_vec[2])
		
		if sent_word in dict_term_sentiment_matrix:
			pos = max(pos, dict_term_sentiment_matrix[sent_word][0])
			neg = max(neg, dict_term_sentiment_matrix[sent_word][1])
		
		neu = 1 if pos == 0 and neg == 0 else 0

		dict_term_sentiment_matrix[sent_word] = (pos, neg, neu)
	
	# Iterate through the vocab of all the documents
	for i, word in vocab_df.iterrows():
		word = word[0]
		print(i, word)
		found = word in dict_term_sentiment_matrix
		if found:
			pos = dict_term_sentiment_matrix[word][0]
			neg = dict_term_sentiment_matrix[word][1]
			neu = dict_term_sentiment_matrix[word][2]
			num_found += 1
		else:
			pos = 0.
			neg = 0.
			neu = 1.

		word_vec = [word, pos, neg, neu]
		print(word_vec)
		print(found)
		doc_term_sentiment_matrix.append(word_vec)
		
	doc_term_sentiment_matrix.append([num_found])
	pd.DataFrame(doc_term_sentiment_matrix).to_csv('doc_term_sentiment_matrix.csv')

def term_sentiment_matrix_to_dataframe(term_sentiment_file):
	term_sentiment_df = pd.read_csv(term_sentiment_file, index_col=0)
	term_sentiment_df = term_sentiment_df.drop('0', 1)
	term_sentiment_df.drop(term_sentiment_df.tail(1).index,inplace=True)
	return term_sentiment_df	

def preprocess_term_sentiment_matrix(term_sentiment_df):
	new_df = pd.DataFrame()
	for _, sent_vec in term_sentiment_df.iterrows():	
		pos = 1 if sent_vec[0] > sent_vec[1] else 0
		neg = 1 if sent_vec[0] < sent_vec[1] else 0
		neu = 1 if sent_vec[0] == sent_vec[1] else 0
		temp_vec = [[pos, neg, neu]]
		new_df = new_df.append(temp_vec)
	return new_df
	
def term_sentiment_matrix_to_context_matrix(term_sentiment_file, preprocess=False, method='tra'):
	term_sentiment_df = term_sentiment_matrix_to_dataframe(term_sentiment_file)
	num_words = term_sentiment_df.shape[0]
	
	if preprocess:
		term_sentiment_df = preprocess_term_sentiment_matrix(term_sentiment_df)
		
	if method == 'tra':
		# Transpose method
		context_matrix = np.dot(term_sentiment_df, np.transpose(term_sentiment_df))
	elif method == 'cos':
		context_matrix = cosine_similarity(term_sentiment_df)
	
	return context_matrix

if __name__ == '__main__':
	#df['RATING'].to_csv("dataset_LABEL.csv", index=False, header=False)
	# output to dataset
	"""print("save bow...")
	df, X = file_to_bow(sys.argv[1])
	scipy.io.savemat(sys.argv[1]+"_bow.mat", {'X' : X})
	print("save tf-idf...")
	df, X = file_to_tfidf(sys.argv[1])
	scipy.io.savemat(sys.argv[1]+"_tf-idf.mat", {'X' : X})
	print("save tf-idf with l2...")
	df, X = file_to_tfidf_l2(sys.argv[1])
	scipy.io.savemat(sys.argv[1]+"_tf-idf-l2.mat", {'X' : X})"""

	# output to vocab file
	"""df, X = file_to_tfidf_l2(sys.argv[1])
	columns = df.columns.values[:-3] # to remove RATING, AUTHOR and DOC_ID columns
	pd.DataFrame(columns).to_csv(sys.argv[1][:-4]+"_vocab.csv", index=False, header=False)"""

	# vocab file to term/sentiment matrix
	"""vocab_to_term_sentiment_matrix(sys.argv[1], sys.argv[2])"""

	# term/sentiment matrix to context matrix
	context_matrix_tra = term_sentiment_matrix_to_context_matrix(sys.argv[1])
	context_matrix_cos = term_sentiment_matrix_to_context_matrix(sys.argv[1], method='cos')
