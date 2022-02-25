import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

def create_tfidf_features(corpus, max_features=2000000, max_df=0.97, min_df=1):
    
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                       ngram_range=(1, 3), max_features=max_features,
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)
    X = tfidf_vectorizor.fit_transform(corpus)
    return X, tfidf_vectorizor

def calculate_similarity(X, vectorizor, query, top_k=5):    
    query_vec = vectorizor.transform(query)
    cosine_similarities = cosine_similarity(X,query_vec).flatten()    
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)

def show_similar_documents(df, cosine_similarities, similar_doc_indices):
    # counter = 1
    for index in similar_doc_indices:
        if cosine_similarities[index] > 0.60:
            return str(cosine_similarities[index]),str(index), str(df['Text'][index])
        else:
            return '','',''