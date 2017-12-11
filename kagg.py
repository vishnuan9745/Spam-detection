from sklearn.cluster import KMeans
import numpy as np
import re
import nltk
import csv
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def reading():
    all_messages = []
    with open("Documents.csv","r") as csvfile:
        myfile = csv.reader(csvfile)
        for line in myfile:
            all_messages.append(line[1])

    # all_messages = all_messages[:]
    print(len(all_messages))
    return all_messages

def clean(msg):
    tokenized_document = []
    vocab_glob = {}
    for tweet in msg:

        tweet = tweet.replace("</p>", "")  # removing </p>
        tweet = tweet.replace("<p>", " ")  # removing <p>
        tweet = tweet.replace("http", " ")
        tweet = tweet.replace("www", " ")
        tweet = re.sub("[^a-zA-Z]", " ", tweet)
        tweet = re.sub('\s+', ' ', tweet)
        tweet = re.sub('\.+', '.', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)  # delete punctuations
        # tweet = re.sub(r"(?:\@\'https?\://)\s+", "", tweet)  # customized deletion
        tweet = re.sub(r'\bRT\b\s+', "", tweet)
        tweet = re.sub("\d+", "", tweet)  # remove number from tweet
        tokens_next = nltk.word_tokenize(tweet)

        stopwords = nltk.corpus.stopwords.words('english')  # stopword reductio
        tokens_next = [w for w in tokens_next if w.lower() not in stopwords and len(w)>2]
        p = PorterStemmer()  # stemming tokenized documents using Porter Stemmer
        # lem = WordNetLemmatizer()
        # tokens_next = [lem.lemmatize(w) for w in tokens_next]
        token_ind = []
        tokens_next = [p.stem(w) for w in tokens_next]

          # adding tokens into global vocabulary
        counter = len(vocab_glob) - 1
        for token in tokens_next:
            if token not in vocab_glob:
                counter += 1
                vocab_glob[token] = counter
                token_ind.append(counter)
            else:
                token_ind.append(vocab_glob[token])

        tokenized_document.append(tokens_next)
    # print("@@@@@@@@@@@@@@@@@@@tokenised document@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(tokenized_document)
    # print("________________________vocab_global___________________")
    # print (vocab_glob)
    return tokenized_document

def feature_selection(final_documents):
    doc_freq = {}
    print "final_documents............"
    print final_documents
    for document in final_documents:
       for ind in document:
         if ind not in doc_freq:
            doc_freq[ind] = 1
         else:
            doc_freq[ind] += 1


   #  print("doc frewwwwww")
   #  print(doc_freq.values())
    top_features = []
    for token in doc_freq.keys():
        if doc_freq[token] >11:    #removing low frequent words(Band Pass Filtering)

            top_features.append(token)
            i = 0
    # print "top words......"
    # print top_features
    top_words = {}
    for token in top_features:
        top_words[i] = token
        i += 1
    print top_words
    return top_words


def clustering(X):

    kmeans = KMeans(n_clusters=2,random_state=0,init='k-means++', n_init=2, tol=0.00001, copy_x=True) .fit(X)
    labels = kmeans.labels_
    print (labels)
    with open("results1.csv","wb") as csvfile:
        myfile=csv.writer(csvfile,dialect="excel")
        myfile.writerow(["SMS_id","label"])
        for index,label in enumerate(labels):
            if index != 0:
                myfile.writerow([index,label])



def feature_matrix_tfidf(top_words,token_ind):


    indexes_features = top_words.values()
    #print indexes_features
    # rows = []
    rows = indexes_features
    columns = []


    for val in token_ind:

        feature_vector = [0] * (len(indexes_features))
        for j in val:
            if j in rows:
                feature_vector[rows.index(j)] = val.count(j)
        columns.append(feature_vector)


    return columns


def feature_vectors(allmsg):
    n_clusters=2
    vec = TfidfTransformer(norm=False,use_idf=True,sublinear_tf=True, smooth_idf=True)
    vector=vec.fit(allmsg)
    # vectorizer=vectorizer.vocabulary_
    vector = vector.transform(allmsg)
    # print(vector.shape)
    # print(type(vector))
    print(vector.toarray())
    #count vectorizer
    return vector

def main():

    all_messages = reading()
    token_ind=clean(all_messages)
    top_words = feature_selection(token_ind)
    final_matrix=feature_matrix_tfidf(top_words,token_ind)
    X = feature_vectors(final_matrix)
    clustering(X)



main()