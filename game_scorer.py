import math
import multiprocessing
import time
from functools import partial
import numpy as np
import nltk
import pandas as pd
import tqdm
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)


def convert_text(text):
    # Importing libraries to work with pandarallel
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    import string
    from nltk.stem.porter import PorterStemmer

    ps = PorterStemmer()

    text = text.lower()
    text = nltk.word_tokenize(text)

    x = []
    y = []
    for i in text:
        try:
            if i not in stopwords.words('english') and i not in string.punctuation:
                x.append(i)
        except Exception as e:
            print("Error: ", e)
    for i in x:
        y.append(ps.stem(i))
    return ' '.join(y)


def execute_sentiment_analysis(text, tfidf, sa_algorithm):
    if text is None or text == "" or (math.isnan(text) if type(text) == float else False):
        return 0
    cv_text = convert_text(text)
    X = tfidf.transform([cv_text])
    return sa_algorithm.predict(X)[0] * max(sa_algorithm.predict_proba(X)[0])


def execute_sentiment_analysis_text_transformed(transformed_text, tfidf, sa_algorithm):
    if transformed_text is None or transformed_text == "" or (
            math.isnan(transformed_text) if type(transformed_text) == float else False):
        return 0
    X = tfidf.transform([transformed_text])
    return sa_algorithm.predict(X)[0] * max(sa_algorithm.predict_proba(X)[0])


def process_row(args, tf_idf, sa_algorithm):
    idx, review_row = args
    # Weight formula: (valoreSentiment x 1,5)(se review_text non null) + 1,3 x review_score + 1,15 x (review_votes)
    game_id = review_row['app_id']
    review_value = review_row['review_score']
    review_votes = review_value * (review_row['review_votes'])
    probability = execute_sentiment_analysis_text_transformed(review_row['converted_text'], tf_idf, sa_algorithm)
    review_score = review_value + (1 + 0.1 * review_votes) * probability

    return {'game_id': game_id, 'score': review_score}


def transform_data(sample, voc_index, s_a_algorithm):
    res = process_row(sample, voc_index, s_a_algorithm)
    thread_score = pd.DataFrame([res])
    # if res['game_id'] in thread_score['game_id'].values:
    #     Update the 'score' value for the existing 'game_id'
    # thread_score.loc[thread_score['game_id'] == res['game_id'], 'score'] += res['score']
    # else:
    #     Create a new DataFrame for the row to append
    #     Concatenate the new DataFrame with the existing 'score' DataFrame
    # thread_score = thread_score.append({'game_id': [res['game_id']], 'score': [res['score']]})
    return thread_score


if __name__ == '__main__':
    print("start")
    revs_og = pd.read_csv('kaggle/dataset_text_converted.csv')
    print("Dataset imported")

    revs_og.dropna(inplace=True)
    # revs = revs_og[['app_id', 'app_name', 'review_score', 'review_text']]
    revs = revs_og
    # revs.dropna(inplace=True)
    new_df = revs
    new_df.reset_index(drop=True, inplace=True)
    # new_df.drop_duplicates(inplace=True)
    # new_df.rename(columns={'review_score': 'target', 'review_text': 'text'}, inplace=True)
    # new_df['char_num'] = new_df['text'].apply(len)
    # new_df['word_num'] = new_df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    # new_df['sent_num'] = new_df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

    # print("beginning text convertion")
    # pandarallel.initialize(nb_workers=12)
    # new_df['converted_text'] = new_df['review_text'].parallel_apply(convert_text)
    # print("text convertion done")
    # print("Saving the converted text")
    # new_df.to_csv('dataset_converted.csv', index=False)
    # print("Done saving")
    print("Starting tfidf")
    tfidf = TfidfVectorizer(max_features=400)
    X_tfid = tfidf.fit_transform(new_df['converted_text']).toarray()
    print("tfidf Done")
    X = X_tfid
    y = new_df['review_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

    sgd = SGDClassifier(loss="modified_huber")
    # sgd.fit(X_train, y_train)

    print("Start Stochastic Gradient Descent training.")
    # Wrap the SGDClassifier training loop with tqdm to create a progress bar
    n_iterations = 100
    with tqdm.tqdm(total=n_iterations, desc="Training Progress") as pbar:
        for epoch in range(n_iterations):
            # Perform one epoch of training (you may need to adjust this part)
            sgd.partial_fit(X_train, y_train, classes=np.unique(y_train))

            # Update the progress bar
            pbar.update(1)
    print("Stochastic Gradient Descent fitted.")

    chosen_algorithm = sgd

    print("Starting multiprocessing game scoring.")

    pred = sgd.predict(X_test)

    print("F1 %", round(f1_score(y_test, pred) * 100, 2))
    print("Precision %", round(precision_score(y_test, pred) * 100, 2))
    print("Recall %", round(recall_score(y_test, pred) * 100, 2))
    print("Accuracy %", round(accuracy_score(y_test, pred) * 100, 2))
    print("Confusion Matrix", confusion_matrix(y_test, pred))
    input("Delay")
    # Create a thread for the first phase
    data_set_used = new_df

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    res_list = []
    # Use partial to create a function with fixed arguments (voc_index and s_a_algorithm)
    transform_data_partial = partial(transform_data, voc_index=tfidf, s_a_algorithm=sgd)


    # Define a function to track the progress
    def track_progress(iterator, total, desc="Processing"):
        with tqdm.tqdm(total=total, desc=desc) as pbar:
            counter = 0
            for item in iterator:
                res_list.append(item)
                yield item
                counter += 1
                if counter == 2000:
                    pbar.update(2000)
                    counter = 0


    start_time = time.time()
    try:
        # Parallelize the execution of transform_data for all samples in data_set_used
        thread_scores_list = pool.imap_unordered(transform_data_partial, data_set_used.iterrows(), chunksize=200)
        # Iterate through the wrapped iterator to trigger progress tracking
        for _ in track_progress(thread_scores_list, total=len(data_set_used), desc="Processing"):
            pass
        # Close the multiprocessing pool
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("Interrupted code execution by the user")
    finally:
        print("Beginning file save")
        # print(res_list)

        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Print the elapsed time
        print(f"Time taken: {elapsed_time} seconds\nProcessed elements = ", (len(data_set_used) / elapsed_time), "/s")

        # Combine individual thread_score DataFrames into a single DataFrame
        # First transform the result into a list.

        # Combining the dataframe
        combined_df = pd.concat(res_list, ignore_index=True)

        # Sum rows with the same game_id to have the total score of a game:
        scores = combined_df.groupby('game_id').sum().reset_index()

        # Save the combined_thread_score DataFrame to CSV
        print("Saving")
        # scores.to_csv('games_score.csv', index=False)
        print("Saved", len(scores), "elements")
        # print(scores)
