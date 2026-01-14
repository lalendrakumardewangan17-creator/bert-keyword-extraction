from sklearn.feature_extraction.text import CountVectorizer

def generate_candidates(text):
    vectorizer = CountVectorizer(ngram_range=(1,3), stop_words='english')
    vectorizer.fit([text])
    return vectorizer.get_feature_names_out()

