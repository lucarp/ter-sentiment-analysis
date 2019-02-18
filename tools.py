import sys
import pandas as pd
import scipy
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
	
print("save bow...")
df, X = file_to_bow(sys.argv[1])
scipy.io.savemat(sys.argv[1]+"_bow.mat", {'X' : X})
print("save tf-idf...")
df, X = file_to_tfidf(sys.argv[1])
scipy.io.savemat(sys.argv[1]+"_tf-idf.mat", {'X' : X})
print("save tf-idf with l2...")
df, X = file_to_tfidf_l2(sys.argv[1])
scipy.io.savemat(sys.argv[1]+"_tf-idf-l2.mat", {'X' : X})

#df['RATING'].to_csv("dataset_LABEL.csv", index=False, header=False)
