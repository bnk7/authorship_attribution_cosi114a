import pandas as pd
from collections import Counter
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

''' Read and clean data '''
df = pd.read_csv("quotes.csv")
# remove tags column
df = df[['author', 'quote']]
# remove book title from author field
df['author'] = df['author'].replace(to_replace=r",.+", value="", regex=True)
# filter to only include data from the top 10 most frequent authors, excluding Anonymous
top_10_authors = df[df.author != 'Anonymous'].author.value_counts().head(10).index
top_10_authors = list(top_10_authors)
df = df[df.author.isin(top_10_authors)]
print('Top 10 authors by number of quotes: ')
print(df.author.value_counts())

# split data into train, dev, and test
x = df.quote
y = df.author
# set the random state in order to get the same split every time
x_train_dev, x_test, y_train_dev, y_test = train_test_split(x, y, test_size=0.1, random_state=3)
dev_size = (1000/90)/100  # to make the dev set 10% of the entire dataset
x_train, x_dev, y_train, y_dev = train_test_split(x_train_dev, y_train_dev, test_size=dev_size, random_state=3)

'''Create features'''
# create "bag of words" features
vectorizer_bow = CountVectorizer(tokenizer=word_tokenize)
X_bow = vectorizer_bow.fit_transform(x_train)
dev_vectorizer_bow = CountVectorizer(vocabulary=vectorizer_bow.get_feature_names_out(), tokenizer=word_tokenize)
dev_X_bow = dev_vectorizer_bow.fit_transform(x_dev)

# create "bag of bigrams" features
vectorizer_bob = CountVectorizer(tokenizer=word_tokenize, ngram_range=(2, 2))
X_bob = vectorizer_bob.fit_transform(x_train)
dev_vectorizer_bob = CountVectorizer(vocabulary=vectorizer_bob.get_feature_names_out(), tokenizer=word_tokenize,
                                     ngram_range=(2, 2))
dev_X_bob = dev_vectorizer_bob.fit_transform(x_dev)

# create tf-idf features
vectorizer_tfidf = TfidfVectorizer(tokenizer=word_tokenize)
X_tfidf = vectorizer_tfidf.fit_transform(x_train)
dev_vectorizer_tfidf = TfidfVectorizer(vocabulary=vectorizer_tfidf.get_feature_names_out(),
                                       tokenizer=word_tokenize)
dev_X_tfidf = dev_vectorizer_tfidf.fit_transform(x_dev)

# create McGill word list feature
# get the 20 tokens that McGill uses more than the other authors
# separate McGill quotes from training data
train_df = pd.DataFrame({'author': y_train, 'quote': x_train})
mcgill_df = train_df[train_df.author == 'Bryant McGill']
num_mcgill = mcgill_df.size
# count tokens in McGill training quotes
mcgill_counts = Counter()
for quote in mcgill_df.quote:
    mcgill_counts.update(word_tokenize(quote.lower()))
# count tokens in other training quotes
other_df = train_df[train_df.author != 'Bryant McGill']
num_other_quotes = other_df.size
other_counts = Counter()
for quote in other_df.quote:
    other_counts.update(word_tokenize(quote.lower()))
# use counts to get average use of each token across quotes
# calculate the difference in averages for tokens that appear in both McGill quotes and other quotes
averages = {key: other_counts[key]/num_other_quotes for key in other_counts}
difs = {key: (mcgill_counts[key]/num_mcgill)-averages[key] for key in mcgill_counts if key in averages}
# get the 20 tokens with the largest difference
difs_series = pd.Series(difs)
difs_series.sort_values(ascending=False, inplace=True)
mcgill_word_set = set(difs_series.head(20).index)

# combine word list feature with bag of bigrams
feat_union_wl_bob = FeatureUnion(
    [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
     ('bob', CountVectorizer(tokenizer=word_tokenize, ngram_range=(2, 2)))])
X_wl_bob = feat_union_wl_bob.fit_transform(x_train)
dev_feat_union_wl_bob = FeatureUnion(
    [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
     ('bob', CountVectorizer(tokenizer=word_tokenize, vocabulary=vectorizer_bob.get_feature_names_out(),
                             ngram_range=(2, 2)))])
dev_X_wl_bob = dev_feat_union_wl_bob.fit_transform(x_dev)

# combine word list feature with tf-idf
feat_union_wl_tfidf = FeatureUnion(
    [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
     ('tfidf', TfidfVectorizer(tokenizer=word_tokenize))])
X_wl_tfidf = feat_union_wl_tfidf.fit_transform(x_train)
dev_feat_union_wl_tfidf = FeatureUnion(
    [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
     ('tfidf', TfidfVectorizer(tokenizer=word_tokenize, vocabulary=vectorizer_tfidf.get_feature_names_out()))])
dev_X_wl_tfidf = dev_feat_union_wl_tfidf.fit_transform(x_dev)

'''Train and test on dev set'''
print('\nAccuracy on dev set with default hyperparameters:')

# Logistic regression
print('Logistic regression')
# scale numbers for better regression
scaledX_bow = MaxAbsScaler().fit_transform(X_bow)
dev_scaledX_bow = MaxAbsScaler().fit_transform(dev_X_bow)
scaledX_bob = MaxAbsScaler().fit_transform(X_bob)
dev_scaledX_bob = MaxAbsScaler().fit_transform(dev_X_bob)
scaledX_tfidf = MaxAbsScaler().fit_transform(X_tfidf)
dev_scaledX_tfidf = MaxAbsScaler().fit_transform(dev_X_tfidf)
scaledX_wl_bob = MaxAbsScaler().fit_transform(X_wl_bob)
dev_scaledX_wl_bob = MaxAbsScaler().fit_transform(dev_X_wl_bob)
scaledX_wl_tfidf = MaxAbsScaler().fit_transform(X_wl_tfidf)
dev_scaledX_wl_tfidf = MaxAbsScaler().fit_transform(dev_X_wl_tfidf)

# train and test using bag of unigrams
logreg = LogisticRegression(solver='saga', max_iter=1000)
logreg.fit(scaledX_bow, y_train)
y_pred_lg_bow = logreg.predict(dev_scaledX_bow)
print('Bag of words: ' + str(accuracy_score(y_dev, y_pred_lg_bow)))

# train and test bag of bigrams
logreg.fit(scaledX_bob, y_train)
y_pred_lg_bob = logreg.predict(dev_scaledX_bob)
print('Bag of bigrams: ' + str(accuracy_score(y_dev, y_pred_lg_bob)))

# train and test tf-idf
logreg_tfidf1 = LogisticRegression(solver='saga', max_iter=1000)
logreg_tfidf1.fit(scaledX_tfidf, y_train)
y_pred_lg_tfidf = logreg_tfidf1.predict(dev_scaledX_tfidf)
print('Tf-idf: ' + str(accuracy_score(y_dev, y_pred_lg_tfidf)))

# train and test bag of bigrams + McGill word list
logreg.fit(scaledX_wl_bob, y_train)
y_pred_lg_wl_bob = logreg.predict(dev_scaledX_wl_bob)
print('Bag of bigrams + McGill word list: ' + str(accuracy_score(y_dev, y_pred_lg_wl_bob)))

# train and test tf-idf + McGill word list
logreg_wl_tfidf1 = LogisticRegression(solver='saga', max_iter=1000)
logreg_wl_tfidf1.fit(scaledX_wl_tfidf, y_train)
y_pred_lg_wl_tfidf = logreg_wl_tfidf1.predict(dev_scaledX_wl_tfidf)
print('Tf-idf + McGill word list: ' + str(accuracy_score(y_dev, y_pred_lg_wl_tfidf)))

# Random forest
print('Random forest')
clf = RandomForestClassifier()

# train and test bag of unigrams
clf.fit(X_bow, y_train)
y_pred_rf_bow = clf.predict(dev_X_bow)
print('Bag of words: ' + str(accuracy_score(y_dev, y_pred_rf_bow)))

# train and test bag of bigrams
clf.fit(X_bob, y_train)
y_pred_rf_bob = clf.predict(dev_X_bob)
print('Bag of bigrams: ' + str(accuracy_score(y_dev, y_pred_rf_bob)))

# train and test tf-idf
clf.fit(X_tfidf, y_train)
y_pred_rf_tfidf = clf.predict(dev_X_tfidf)
print('Tf-idf: ' + str(accuracy_score(y_dev, y_pred_rf_tfidf)))

# train and test bag of bigrams + McGill word list
clf.fit(X_wl_bob, y_train)
y_pred_rf_wl_bob = clf.predict(dev_X_wl_bob)
print('Bag of bigrams + McGill word list: ' + str(accuracy_score(y_dev, y_pred_rf_wl_bob)))

# train and test tf-idf + McGill word list
clf.fit(X_wl_tfidf, y_train)
y_pred_rf_wl_tfidf = clf.predict(dev_X_wl_tfidf)
print('Tf-idf + McGill word list: ' + str(accuracy_score(y_dev, y_pred_rf_wl_tfidf)))

'''Tune'''
print('\nAccuracy on dev set, tuning hyperparameters:')
print('Logistic regression')
logreg05 = LogisticRegression(solver='saga', max_iter=1000, C=0.5)
logreg05.fit(scaledX_bow, y_train)
y_pred_lg_bow = logreg05.predict(dev_scaledX_bow)
print('Bag of words, C=0.5: ' + str(accuracy_score(y_dev, y_pred_lg_bow)))

logreg2 = LogisticRegression(solver='saga', max_iter=1000, C=2.0)
logreg2.fit(scaledX_bow, y_train)
y_pred_lg_bow = logreg2.predict(dev_scaledX_bow)
print('Bag of words, C=2.0: ' + str(accuracy_score(y_dev, y_pred_lg_bow)))

logreg05.fit(scaledX_tfidf, y_train)
y_pred_lg_tfidf = logreg05.predict(dev_scaledX_tfidf)
print('Tf-idf, C=0.5: ' + str(accuracy_score(y_dev, y_pred_lg_tfidf)))

logreg_tfidf2 = LogisticRegression(solver='saga', max_iter=1000, C=2.0)
logreg_tfidf2.fit(scaledX_tfidf, y_train)
y_pred_lg_tfidf = logreg_tfidf2.predict(dev_scaledX_tfidf)
print('Tf-idf, C=2.0: ' + str(accuracy_score(y_dev, y_pred_lg_tfidf)))

logreg05.fit(scaledX_wl_tfidf, y_train)
y_pred_lg_wl_tfidf = logreg05.predict(dev_scaledX_wl_tfidf)
print('Tf-idf + McGill word list, C=0.5: ' + str(accuracy_score(y_dev, y_pred_lg_wl_tfidf)))

logreg2.fit(scaledX_wl_tfidf, y_train)
y_pred_lg_wl_tfidf = logreg2.predict(dev_scaledX_wl_tfidf)
print('Tf-idf + McGill word list, C=2.0: ' + str(accuracy_score(y_dev, y_pred_lg_wl_tfidf)))

clf = RandomForestClassifier(n_estimators=50)
print('Random forest')
clf.fit(X_tfidf, y_train)
y_pred_rf_tfidf = clf.predict(dev_X_tfidf)
print('Tf-idf, n_estimators=50, max_features=sqrt: ' + str(accuracy_score(y_dev, y_pred_rf_tfidf)))

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_tfidf, y_train)
y_pred_rf_tfidf = clf.predict(dev_X_tfidf)
print('Tf-idf, n_estimators=200, max_features=sqrt: ' + str(accuracy_score(y_dev, y_pred_rf_tfidf)))

clf = RandomForestClassifier(n_estimators=50, max_features=1.0)
clf.fit(X_tfidf, y_train)
y_pred_rf_tfidf = clf.predict(dev_X_tfidf)
print('Tf-idf, n_estimators=50, max_features=1.0: ' + str(accuracy_score(y_dev, y_pred_rf_tfidf)))

clf = RandomForestClassifier(max_features=1.0)
clf.fit(X_tfidf, y_train)
y_pred_rf_tfidf = clf.predict(dev_X_tfidf)
print('Tf-idf, n_estimators=100, max_features=1.0: ' + str(accuracy_score(y_dev, y_pred_rf_tfidf)))

'''Test'''
# create tf-idf and word list features for test set
test_X_tfidf = dev_vectorizer_tfidf.fit_transform(x_test)
test_X_wl_tfidf = dev_feat_union_wl_tfidf.fit_transform(x_test)
# scale
test_scaledX_tfidf = MaxAbsScaler().fit_transform(test_X_tfidf)
test_scaledX_wl_tfidf = MaxAbsScaler().fit_transform(test_X_wl_tfidf)
# test
print('\nAccuracy on test set, logistic regression only:')
y_pred_lg_tfidf1 = logreg_tfidf1.predict(test_scaledX_tfidf)
print('Tf-idf, C=1.0: ' + str(accuracy_score(y_test, y_pred_lg_tfidf1)))
y_pred_lg_tfidf2 = logreg_tfidf2.predict(test_scaledX_tfidf)
print('Tf-idf, C=2.0: ' + str(accuracy_score(y_test, y_pred_lg_tfidf2)))
y_pred_lg_wl_tfidf = logreg_wl_tfidf1.predict(test_scaledX_wl_tfidf)
print('Tf-idf + McGill word list, C=1.0: ' + str(accuracy_score(y_test, y_pred_lg_wl_tfidf)))
print('\nClassification report for logistic regression, tf-idf, C=2.0:')
print(classification_report(y_test, y_pred_lg_tfidf2))
