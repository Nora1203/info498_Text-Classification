# Project 2: Text Classification

**Objectives:** To train text classification models and analyze their results.

**What to turn in:** A jupyter notebook containing all your code and results. When asked to discuss results, create a Markdown cell for your text. 

<!-- In some cases, I ask for graphs to illustrate distributions and modeling results. You can create these using python libraries (matplotlib, seaborn) if you are familiar and comfortable using these libraries (not required), or if not, you can use other tools like Excel or Google Sheets to produce graphs, and embed them in the notebook in a Markdown cell (Edit -> Insert Image). If you use python to make plots, you can modify you environment to include matplotlib and seaborn, and you can assume I have access to these libraries in my environment as well. If you make graphs elsewhere and embed them, make sure you upload those images when submitting your assignment as well. -->

Please name the file `<uwnetid>-project2.ipynb`, replacing `<uwnetid>` with your UW net ID. Turn the file in via Canvas.

All commands in the notebook should be able to run assuming the data files are located under the relative path `data/`. Label cell for each problem, and make sure I can easily find where your answers are and what they are! At least 1-2 points for each question will be awarded for style.

**Due date:** Thursday, October 26, 2023 EOD

In the `data/` subdirectory, you will find several directories. Each subdirectory corresponds to a specific dataset.

Some starter code is provided to you in `demo.ipynb`.

**Total points:** Graded out of 100 points (120 total points available)

## Train a logistic regression model for binary classification

You will use the Twitter airline tweets sentiment classification dataset to train this model. The dataset is located under `data/twitter_binary_classification_dataset/`. This directory contains two files: `train.csv` and `test.csv`. The tweet text is under the column `text` and the sentiment label is under the column `airline_sentiment`. 

1. (5 pt) Load both the train and test splits and compute summary statistics for the dataset:
    * How many tweets are there? 
    * What is the timeframe associated with these tweets (earliest and latest tweet)? 
    * What is the unique set of airlines that are represented? 
    * What are the numbers of tweets per airline?
    * How many tweets are associated with each sentiment label?

2. (5 pt) The sentiment labels are positive, negative, and neutral. Produce a binary classification dataset by removing all tweet instances that are neutral. How does this change the dataset distribution (tweets per airline, tweets per label)?

3. (5 pt) Convert the tweet texts into feature vectors. You should:
    * Remove the "@airline" token from all tweets
    * Then use the tf-idf vectorizer to convert all texts into weighted vectors

4. (5 pt) Convert the labels into 1s (positive sentiment) and 0s (negative sentiment).

5. (10 pt) Train a logistic regression classifier only on the vectors for data from the **train** split. Report the performance of the classifier on the **train** split. Compute precision, recall, f1-score, and accuracy.

6. (10 pt) Now use your trained classifier to perform inference on the **test** split. Report performance on the **test** split. Compute precision, recall, f1-score, and accuracy. Discuss the performance differences between the train and test splits.

7. (10 pt) Compute and show the confusion matrix on the **test** split. Discuss the performance differences between classes and how you could potentially mitigate any differences you observe.

## Implement a KNN model for multi-class classification using BERT document embeddings

You will use the AG news multi-class classification dataset to train this model. The dataset is located under `data/ag_news_multiclass_classification_dataset/`. This directory contains two files: `train.csv` and `test.csv`. Each file is a three-column csv, where the columns correspond to `label`, `title`, and `text`. The label corresponds to the class as named in the `classes.txt` file. For each row, you should concatenate the title and text together (make sure you add a space) into a single news article (the document).

8. (5 pt) Load both the train and test splits and compute summary statistics for the dataset (by split):
    * How many news articles are in each split?
    * What is the average, min, and max document length in each split?
    * What is the distribution of labels in each split?

9. (~~10~~ 20 pt) Use the tf-idf vectorizer to convert all texts into weighted vectors; you should fit on the train set and transform the test set. 
    * For each word token, find the highest tf-idf weight among all the **train** set documents that is associated with that token. What are the 10 highest tf-idf weighted tokens from any **train** set document for each label? 
    * Based on these top word tokens: what would you guess are the news topic represented by each of the class labels? Does this match the actual labels from the `classes.txt` file?

10. ~~(10 pt) Create a random subsample of 3000 instances from the **train** split and 1000 instances from the **test** split (using a smaller subsample will make this much more tractable since embedding the documents will take some time). Use Sentence-BERT to embed each document in the subsampled **train** and **test** splits.~~ 

11. (20 pt) For each document in the **test** split, classify the document using KNN on the ~~Sentence-BERT embeddings~~ document tf-idf vectors with k=5 and cosine similarity. Note: you should only be comparing against the documents in the **train** split for classification, not those in the **test** split. Report the performance of your model. Compute micro- and macro-F1.

12. (5 pt) When classifying documents in the **test** split, explain why you should only compare against documents in the **train** split rather than documents in both the **train** and **test** splits.

13. (10 pt) Report the per-class F1-scores for the **test** split. Examine some of the mis-classified documents. Discuss whether you believe these documents have been truly mis-classified and why.

## Comparing classification models

14. (10 pt) For each of the two models you trained (logistic regression and KNN), discuss its pros and cons. Why might someone choose one model architecture over another?
