import sys
import gensim
import pandas as pd

def train_dataset(reviews):
	for idx, item in reviews.iterrows():
		yield gensim.models.doc2vec.TaggedDocument(item['Review'].split(' '), [idx])


def doc2Vec(file_name, vec_size = 50):
	texts_df = pd.read_csv(file_name, header=0, index_col=0)

	num_documents = texts_df.shape[0]

	print("make corpus...")
	train_corpus = list(train_dataset(texts_df))

	model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, epochs = 500)
	model.build_vocab(train_corpus)
	print("training...")
	model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
	
	print("save...")
	vecs = []
	i = 0
	for item in model.docvecs:
		vecs.append(item)
		i +=1
		if i > num_documents - 1:
			break
	
	model.save(file_name+"_doc2vec.model")
	
	return vecs

if __name__ == "__main__":
	vecs = doc2Vec(sys.argv[1])
	pd.DataFrame(vecs).to_csv(sys.argv[1]+"_doc2Vec.csv")
