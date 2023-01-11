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


if __name__ == '__main__':
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
    train_features = {}
    dev_features = {}

    # create "bag of words" features
    vectorizer_bow = CountVectorizer(tokenizer=word_tokenize)
    train_features['bag of words'] = vectorizer_bow.fit_transform(x_train)
    dev_vectorizer_bow = CountVectorizer(vocabulary=vectorizer_bow.get_feature_names_out(), tokenizer=word_tokenize)
    dev_features['bag of words'] = dev_vectorizer_bow.fit_transform(x_dev)

    # create "bag of bigrams" features
    vectorizer_bob = CountVectorizer(tokenizer=word_tokenize, ngram_range=(2, 2))
    train_features['bag of bigrams'] = vectorizer_bob.fit_transform(x_train)
    dev_vectorizer_bob = CountVectorizer(vocabulary=vectorizer_bob.get_feature_names_out(), tokenizer=word_tokenize,
                                         ngram_range=(2, 2))
    dev_features['bag of bigrams'] = dev_vectorizer_bob.fit_transform(x_dev)

    # create tf-idf features
    vectorizer_tfidf = TfidfVectorizer(tokenizer=word_tokenize)
    train_features['tf-idf'] = vectorizer_tfidf.fit_transform(x_train)
    dev_vectorizer_tfidf = TfidfVectorizer(vocabulary=vectorizer_tfidf.get_feature_names_out(),
                                           tokenizer=word_tokenize)
    dev_features['tf-idf'] = dev_vectorizer_tfidf.fit_transform(x_dev)

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
    train_features['word list and bag of bigrams'] = feat_union_wl_bob.fit_transform(x_train)
    dev_feat_union_wl_bob = FeatureUnion(
        [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
         ('bob', CountVectorizer(tokenizer=word_tokenize, vocabulary=vectorizer_bob.get_feature_names_out(),
                                 ngram_range=(2, 2)))])
    dev_features['word list and bag of bigrams'] = dev_feat_union_wl_bob.fit_transform(x_dev)

    # combine word list feature with tf-idf
    feat_union_wl_tfidf = FeatureUnion(
        [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
         ('tfidf', TfidfVectorizer(tokenizer=word_tokenize))])
    train_features['word list and tf-idf'] = feat_union_wl_tfidf.fit_transform(x_train)
    dev_feat_union_wl_tfidf = FeatureUnion(
        [('mcgill_word_count', CountVectorizer(tokenizer=word_tokenize, vocabulary=mcgill_word_set)),
         ('tfidf', TfidfVectorizer(tokenizer=word_tokenize, vocabulary=vectorizer_tfidf.get_feature_names_out()))])
    dev_features['word list and tf-idf'] = dev_feat_union_wl_tfidf.fit_transform(x_dev)

    '''Train and test on dev set'''
    print('\nAccuracy on dev set with default hyperparameters:')

    # Logistic regression
    print('Logistic regression')
    # scale numbers for better regression
    train_scaled_features = {}
    dev_scaled_features = {}
    for feature in train_features:
        train_scaled_features[feature] = MaxAbsScaler().fit_transform(train_features[feature])
        dev_scaled_features[feature] = MaxAbsScaler().fit_transform(dev_features[feature])

    # train and test dev
    logreg = LogisticRegression(solver='saga', max_iter=1000)
    lg_accuracies = {}
    for feature in train_scaled_features:
        logreg.fit(train_scaled_features[feature], y_train)
        y_pred = logreg.predict(dev_scaled_features[feature])
        lg_accuracies[feature] = accuracy_score(y_dev, y_pred)
    for feature in lg_accuracies:
        print(feature + ': ' + str(lg_accuracies[feature]))

    # Random forest
    print('Random forest')
    # train and test dev
    clf = RandomForestClassifier()
    rf_accuracies = {}
    for feature in train_features:
        clf.fit(train_features[feature], y_train)
        y_pred = clf.predict(dev_features[feature])
        rf_accuracies[feature] = accuracy_score(y_dev, y_pred)
    for feature in rf_accuracies:
        print(feature + ': ' + str(rf_accuracies[feature]))

    '''Tune'''
    print('\nAccuracy on dev set, tuning hyperparameters:')
    print('Logistic regression')
    lg_tuning_accuracies = {}
    for c in (0.5, 2.0):
        logreg = LogisticRegression(solver='saga', max_iter=1000, C=c)
        for feature in ('bag of words', 'tf-idf', 'word list and tf-idf'):
            logreg.fit(train_scaled_features[feature], y_train)
            y_pred = logreg.predict(dev_scaled_features[feature])
            lg_tuning_accuracies[(feature, c)] = accuracy_score(y_dev, y_pred)
    for config in lg_tuning_accuracies:
        print(config[0] + ', C=' + str(config[1]) + ': ' + str(lg_tuning_accuracies[config]))

    print('Random forest')
    rf_tuning_accuracies = {}
    # max_features = sqrt
    for n in (50, 200):
        clf = RandomForestClassifier(n_estimators=n, max_features='sqrt')
        clf.fit(train_features['tf-idf'], y_train)
        y_pred = clf.predict(dev_features['tf-idf'])
        rf_tuning_accuracies[(n, 'sqrt')] = accuracy_score(y_dev, y_pred)
    # max_features = 1.0
    for n in (50, 100):
        clf = RandomForestClassifier(n_estimators=n, max_features=1.0)
        clf.fit(train_features['tf-idf'], y_train)
        y_pred = clf.predict(dev_features['tf-idf'])
        rf_tuning_accuracies[(n, 1.0)] = accuracy_score(y_dev, y_pred)
    for config in rf_tuning_accuracies:
        print('tf-idf, n_estimators=' + str(config[0]) + ', max_features=' + str(config[1]) + ': ' +
              str(rf_tuning_accuracies[config]))

    '''Test'''
    # create tf-idf and word list features for test set
    test_X_tfidf = dev_vectorizer_tfidf.fit_transform(x_test)
    test_X_wl_tfidf = dev_feat_union_wl_tfidf.fit_transform(x_test)
    # scale
    test_scaledX_tfidf = MaxAbsScaler().fit_transform(test_X_tfidf)
    test_scaledX_wl_tfidf = MaxAbsScaler().fit_transform(test_X_wl_tfidf)
    # test
    print('\nAccuracy on test set, logistic regression only:')

    logreg = LogisticRegression(solver='saga', max_iter=1000)
    logreg.fit(train_scaled_features['tf-idf'], y_train)
    y_pred_lg_tfidf1 = logreg.predict(test_scaledX_tfidf)
    print('tf-idf, C=1.0: ' + str(accuracy_score(y_test, y_pred_lg_tfidf1)))

    logreg2 = LogisticRegression(solver='saga', max_iter=1000, C=2.0)
    logreg2.fit(train_scaled_features['tf-idf'], y_train)
    y_pred_lg_tfidf2 = logreg2.predict(test_scaledX_tfidf)
    print('tf-idf, C=2.0: ' + str(accuracy_score(y_test, y_pred_lg_tfidf2)))

    logreg.fit(train_scaled_features['word list and tf-idf'], y_train)
    y_pred_lg_wl_tfidf = logreg.predict(test_scaledX_wl_tfidf)
    print('word list and tf-idf, C=1.0: ' + str(accuracy_score(y_test, y_pred_lg_wl_tfidf)))

    print('\nClassification report for logistic regression, tf-idf, C=2.0:')
    print(classification_report(y_test, y_pred_lg_tfidf2))
