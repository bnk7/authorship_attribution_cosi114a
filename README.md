# Authorship Attribution of Popular Quotes
Brynna Kilcline
## Introduction
For this project, I was interested to see if fundamental natural language processing
techniques could be used to predict the author of a short quote. I chose this topic due to
my general enjoyment of books and the availability of preprocessed data.
## Data
The dataset is called "Quotes-500k" and comes from Goel, Madhok, and Garg (2018), 
which I found on Kaggle. The authors scraped the quotes from various quote websites, 
filtered out quotes in languages other than English, and created a CSV with the quote,
the author, and its tags.

I preprocessed the data by removing the tags column and normalizing the author column 
to remove book titles. I then filtered the data to only include the 10 authors with 
the highest number of quotes. This resulted in 34,997 data points, divided among the 
10 authors as shown in Table 1.

**Table 1: Most frequent authors**

| Author              | Data Points |
|---------------------|-------------|
| Lailah Gifty Akita  | 10620       |
| Debasish Mridha     | 6601        |
| Sunday Adelaja      | 5891        |
| Matshona Dhliwayo   | 2259        |
| Israelmore Ayivor   | 2079        |
| Billy Graham        | 1953        |
| Mehmet Murat ildan  | 1927        |
| Deyth Banger        | 1249        |
| William Shakespeare | 1248        |
| Bryant McGill       | 1170        |

Here are some examples of the data. I did not alter the quotes in any way.

**Table 2: Example data**

| Author             | Quote                                                                                                            |
|--------------------|------------------------------------------------------------------------------------------------------------------|
| Lailah Gifty Akita | You can courageously conquer any challenge.                                                                      |
| Debasish Mridha    | A writer fills the paper with the pictures of perception, experience, and feeling of real and imaginative lives. |
| Matshona Dhliwayo  | Storms are rainbow’s inside out.                                                                                 |

I split the data into 80 percent training, 10 percent development, and 10 percent test.
Thus, there are 27,997 instances in the training set and 3,500 instances each in the
development and test sets.

## Results
### Dev Set Results
For this multiclass classification problem, I used two models: multiclass logistic regression 
and random forest. To run the models, I used `LogisticRegression` and `RandomForestClassifier`
from scikit-learn (Pedregosa et al., 2011). I initially used the features of bag of unigrams
and bag of bigrams, extracted from the data using `CountVectorizer`, and tf-idf vectors,
extracted using `TfidfVectorizer`. In order to get the logistic regression to converge,
I scaled the features using `MaxAbsScaler`, set the solver to SAGA, and increased the maximum
number of iterations to 1000. I kept the rest of the parameters at the default value to begin.

After running a few initial tests on the dev set using each of
these features individually, I noticed that the F1 and recall were especially low for the
author Bryant McGill, who is the least represented in the data, particularly for the random 
forest models. So, I decided to add a word list feature for this author. The list comprises 
the top 20 word types that McGill used more frequently than the other authors in the training
data, given that at least one of the other authors used it (my reasoning being that the other
features I had chosen would already account for word types completely unique to McGill). 
I determined it by calculating the average number of 
times each type appeared in a Bryant McGill training quote and the average number of times it
appeared in a non-McGill training quote and finding the difference between the two. Once the
word list was created, I made it into a feature using `CountVectorizer` with the list as
the vocabulary. Then, I combined it with tf-idf and bag of bigrams but not bag of unigrams
because the word list feature is itself a bag of selected unigrams.

Overall, the word list feature did not make a consistent difference in the performance on the 
dev set. The accuracies using the default hyperparameters (C = 1.0 for logistic regression 
and n_estimators = 100, max_features = sqrt for random forest) are shown in Table 3 below.

**Table 3: Accuracy scores for all models and features, pre-tuning**

| Model               | Features                  | Accuracy  |
|---------------------|---------------------------|-----------|
| Logistic Regression | bag of unigrams           | 77.51     |
| Logistic Regression | bag of bigrams            | 75.26     |
| Logistic Regression | tf-idf                    | **80.94** |
| Logistic Regression | bag of bigrams, word list | 75.43     |
| Logistic Regression | tf-idf, word list         | 80.49     |
| Random Forest       | bag of unigrams           | 72.57     |
| Random Forest       | bag of bigrams            | 57.74     |
| Random Forest       | tf-idf                    | 73.31     |
| Random Forest       | bag of bigrams, word list | 60.86     |
| Random Forest       | tf-idf, word list         | 72.63     |

I picked four configurations to tune: the three best overall, which were all logistic
regression models, and the best random forest model. For logistic regression, I tested
C = 0.5, 1.0, and 2.0 for the features bag of unigrams, tf-idf, and tf-idf plus word list.
The results are in Table 4.

For random forest, I tried n_estimators = 50, 100, and 200 and max_features = sqrt and 1.0. 
I tried each possible configuration with the tf-idf feature except n_estimators = 50 combined
with max_features = 1.0 because my computer had a hard time with the required RAM and the
pattern showed it was unlikely to perform as well as max_features = sqrt anyway. The accuracies on
the dev set are in Table 5.

**Table 4: Logistic Regression Tuning**

| Features          | C   | Accuracy  |
|-------------------|-----|-----------|
| bag of unigrams   | 0.5 | 77.09     |
| bag of unigrams   | 1.0 | 77.51     |
| bag of unigrams   | 2.0 | 77.77     |
| tf-idf            | 0.5 | 80.34     |
| tf-idf            | 1.0 | 80.94     |
| tf-idf            | 2.0 | **81.26** |
| tf-idf, word list | 0.5 | 80.00     |
| tf-idf, word list | 1.0 | 80.49     |
| tf-idf, word list | 2.0 | 80.31     |

**Table 5: Random Forest Tf-idf Tuning**

| n_estimators | max_features | Accuracy  |
|--------------|--------------|-----------|
| 50           | sqrt         | 72.03     |
| 100          | sqrt         | 73.31     |
| 200          | sqrt         | **73.74** |
| 50           | 1.0          | 67.06     |
| 100          | 1.0          | 67.63     |

### Test Set Results
I chose the three configurations that performed best on the dev set to run on the
test set. These were logistic regression configurations with (1) tf-idf features
and C=1.0, (2) tf-idf features and C=2.0, and (3) tf-idf and word list features and
C=1.0.

**Table 6: Accuracy scores on test set**

| Model               | Features          | C   | Accuracy  |
|---------------------|-------------------|-----|-----------|
| Logistic regression | tf-idf            | 1.0 | 81.49     |
| Logistic regression | tf-idf            | 2.0 | **81.80** |
| Logistic regression | tf-idf, word list | 1.0 | 80.97     |

The three accuracy scores were close, but the model with a C of 2.0 performed the 
best. More details on this model's performance are in Table 7.

**Table 7: Classification report for logistic regression with tf-idf features and C=2.0**

| Author              | Precision | Recall    | F1-score  | Support |
|---------------------|-----------|-----------|-----------|---------|
| Billy Graham        | 82.08     | 74.00     | 77.81     | 192     |
| Bryant McGill       | 56.45     | 31.53     | 40.46     | 111     |
| Debasish Mridha     | 79.52     | 87.29     | 83.22     | 645     |
| Deyth Banger        | 86.78     | 83.33     | 85.02     | 126     |
| Israelmore Ayivor   | 73.33     | 72.30     | 72.81     | 213     |
| Lailah Gifty Akita  | **89.05** | 84.12     | 86.51     | 1102    |
| Matshona Dhliwayo   | 72.83     | 66.34     | 69.43     | 202     |
| Mehmet Murat ildan  | 82.06     | **91.96** | **86.73** | 199     |
| Sunday Adelaja      | 78.99     | 91.38     | 84.73     | 580     |
| William Shakespeare | 84.11     | 69.23     | 75.95     | 130     |
|                     |           |           |           |         |
| accuracy            |           |           | 81.80     | 3500    |
| macro avg           | 78.52     | 75.11     | 76.27     | 3500    |
| weighted avg        | 81.65     | 81.80     | 81.42     | 3500    |

## Discussion
I was surprised that logistic regression, which is often seen as a baseline model, performed 
better than random forest. That said, it does make sense to have a model that generally 
performs well on different datasets as the standard baseline. I was also surprised that the
word list, which I created with the intention of boosting the performance on the 
poorest-performing class, did not improve scores. A possible explanation is that the quotes 
by Bryant McGill made up a relatively small portion of the data anyway, so attempts focused 
on that category may not make much of a difference on the overall performance. As is to be
expected, the tf-idf features performed better than the bags of n-grams because it gives
higher weight to less frequent terms.

While I was hoping for a higher accuracy, I think the final accuracy of 81 was decent 
for a 10-way classification task based on at most a few sentences at a time.
## Conclusion
Overall, in this project I was able to achieve 81.80 percent accuracy in predicting the author
from a quote and a list of 10 authors using a logistic regression model and tf-idf features.

The hardest part of the process was understanding what format the data had to be in to use
the tools in scikit-learn and shaping the data to that format. If I had more time on this
problem, I would read more about random forest to understand why it might not work as well as
logistic regression and focus on making my code more efficient.
## References
Goel, S., Madhok, R., & Garg, S. (2018). Proposing contextually relevant quotes for images. 
*40th European Conference on Information Retrieval.* 
[https://github.com/ShivaliGoel/Quotes-500K](https://github.com/ShivaliGoel/Quotes-500K)

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011).
Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2,825-2,830.
