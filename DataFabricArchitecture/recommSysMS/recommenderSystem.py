import multiprocessing
from datetime import time
from functools import partial

from lightfm import LightFM
from lightfm.data import Dataset

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from game_scorer import transform_data


#####################
# Functions section #
#####################
# --------------------------------------------------------------------------------------------------#
def startSparkSession(app_name):
    return SparkSession \
        .builder \
        .appName(app_name) \
        .getOrCreate()


def readFromKafkaStreamingTopic(host_port, topic_name):
    try:
        return spark_session \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", host_port) \
            .option("subscribe", topic_name) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
    except Exception as e:
        print(e)


# Define a function to track the progress
def track_progress(iterator, total, desc="Processing"):
    with tqdm(total=total, desc=desc) as pbar:
        counter = 0
        for item in iterator:
            res_list.append(item)
            yield item
            counter += 1
            if counter == 2000:
                pbar.update(2000)
                counter = 0



#####################
# VARIABLES SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    application_name = "recsys-ingestion-util"
    kafka_host_port = "localhost:9092"
    reviews_topic_name = "reviews-ingestion-topic"
    users_topic_name = "users-ingestion-topic"
    tags_topic_name = "tags-ingestion-topic"

    review_schema = StructType([
        StructField("id", IntegerType()),
        StructField("app_id", StringType()),
        StructField("app_name", StringType()),
        StructField("sentiment_target", StringType()),
        StructField("text", StringType()),
        StructField("converted_text", StringType())
    ])

    tag_schema = StructType([
        StructField("app_id", StringType()),
        StructField("categories", ArrayType(StringType()))
    ])

    user_games_schema = StructType([
        StructField("user_id", StringType()),
        StructField("games", ArrayType(StructType([
            StructField("appid", StringType()),
            StructField("playtime_forever", DoubleType()),
            StructField("playtime_windows_forever", DoubleType()),
            StructField("playtime_mac_forever", DoubleType()),
            StructField("playtime_linux_forever", DoubleType()),
            StructField("rtime_last_played", DoubleType()),
            StructField("playtime_disconnected", DoubleType()),
        ])))
    ])


#################
# CHECK SECTION #
#################
# --------------------------------------------------------------------------------------------------#
    spark_session = startSparkSession(application_name)
    reviews_kafkaStreamReader = readFromKafkaStreamingTopic(kafka_host_port, reviews_topic_name)
    tags_kafkaStreamReader = readFromKafkaStreamingTopic(kafka_host_port, tags_topic_name)
    users_kafkaStreamReader = readFromKafkaStreamingTopic(kafka_host_port, users_topic_name)

    reviews_check = 0
    game_tags_check = 0
    user_games_check = 0

    while reviews_check == 0 or game_tags_check == 0 or user_games_check == 0:
        if reviews_check == 0:
            reviews_df = reviews_kafkaStreamReader\
                .withColumn("json", from_json(col("value").cast("string"), review_schema)) \
                .toPandas()
            if reviews_df.size > 0:
                reviews_check = 1

        if game_tags_check == 0:
            game_tags_df = tags_kafkaStreamReader\
                .withColumn("json", from_json(col("value").cast("string"), tag_schema)) \
                .toPandas()
            if game_tags_df.size > 0:
                game_tags_check = 1

        if user_games_check == 0:
            user_games_df = users_kafkaStreamReader \
                .withColumn("json", from_json(col("value").cast("string"), user_games_schema)) \
                .toPandas()
            if user_games_df.size > 0:
                user_games_check = 1


######################
# GameScorer SECTION #
######################
# --------------------------------------------------------------------------------------------------#
    print("Starting tfidf")
    tfidf = TfidfVectorizer(max_features=400)
    X_tfid = tfidf.fit_transform(reviews_df['converted_text']).toarray()
    print("tfidf Done")
    X = X_tfid
    y = reviews_df['review_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

    sgd = SGDClassifier(loss="modified_huber")

    print("Start Stochastic Gradient Descent training.")
    # Wrap the SGDClassifier training loop with tqdm to create a progress bar
    n_iterations = 100
    with tqdm(total=n_iterations, desc="Training Progress") as pbar:
        for epoch in range(n_iterations):
            # Perform one epoch of training (you may need to adjust this part)
            sgd.partial_fit(X_train, y_train, classes=np.unique(y_train))

            # Update the progress bar
            pbar.update(1)
    print("Stochastic Gradient Descent fitted.")

    chosen_algorithm = sgd

    print("Starting multiprocessing game scoring.")

    pred = sgd.predict(X_test)

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    res_list = []
    # Use partial to create a function with fixed arguments (voc_index and s_a_algorithm)
    transform_data_partial = partial(transform_data, voc_index=tfidf, s_a_algorithm=sgd)

    start_time = time()
    try:
        # Parallelize the execution of transform_data for all samples in data_set_used
        thread_scores_list = pool.imap_unordered(transform_data_partial, reviews_df.iterrows(), chunksize=200)
        # Iterate through the wrapped iterator to trigger progress tracking
        for _ in track_progress(thread_scores_list, total=len(reviews_df), desc="Processing"):
            pass
        # Close the multiprocessing pool
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("Interrupted code execution by the user")
    finally:
        print("Beginning file save")

        end_time = time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Print the elapsed time
        print(f"Time taken: {elapsed_time} seconds\nProcessed elements = ", (len(reviews_df) / elapsed_time), "/s")

        # Combine individual thread_score DataFrames into a single DataFrame
        # First transform the result into a list.

        # Combining the dataframe
        combined_df = pd.concat(res_list, ignore_index=True)

        # Sum rows with the same game_id to have the total score of a game:
        scores = combined_df.groupby('game_id').sum().reset_index()




    ######################
    # Categories SECTION #
    ######################
    # --------------------------------------------------------------------------------------------------#
    game_tags_df.reset_index(drop=True, inplace=True)
    game_categories = game_tags_df.melt(var_name="game_id", value_name="categories")
    game_categories = game_categories.dropna()
    game_categories = game_categories.groupby('game_id')['categories'].agg(list).reset_index()
    game_categories['game_id'] = game_categories['game_id'].astype('int64')
    game_categories.reset_index(drop=True, inplace=True)


##################
# RecSys SECTION #
##################
# --------------------------------------------------------------------------------------------------#
    game_id_counts = reviews_df['app_id'].value_counts().reset_index()
    game_id_counts.columns = ['app_id', 'occurrence']
    game_id_counts.rename(columns={'app_id': 'game_id'}, inplace=True)

    game_scores_normalized = pd.merge(scores, game_id_counts, on ='game_id', how='left')

    # Formula used for score normalization, where:
    #   k := sweet spot
    k = 20
    game_scores_normalized['normalized_score'] = \
        (game_scores_normalized['score'] /
         game_scores_normalized['occurrence']) * (0.8 + 0.2 * (game_scores_normalized['occurrence'] /
                                                                   (game_scores_normalized['occurrence'] + k))) / 2 * 10
    game_df = pd.merge(game_scores_normalized, game_categories, on ='game_id', how='inner')
    game_df = game_df[['game_id','normalized_score','categories', 'occurrence', 'score']]
    dataset = Dataset(user_identity_features=False)
    # Fitting the recommender system by giving every used_id and game_id in the datasets.
    dataset.fit(game_categories['user_id'], game_df['game_id'])

    for _, row in game_df.iterrows():
        game_id, _, categories, _, _ = row
        dataset.fit_partial(items=[game_id], item_features=categories)

    game_categories_user_score = pd.merge(game_categories, user_games_df, on="game_id", how="right")
    game_categories_user_score.head()
    exploded_df = game_categories_user_score.explode("categories")
    aggregated_df = exploded_df.groupby(["user_id", "categories"])["playtime_forever"].sum().reset_index()
    user_category_score = aggregated_df.rename(columns={"categories": "category","playtime_forever": "user_category_score"})

    for _, row in user_category_score.iterrows():
        user_id, category, category_score= row
        dataset.fit_partial(users=[user_id], user_features={category: category_score})

    users_features = dataset.build_user_features(((x['user_id'], {x['category']: x['user_category_score']})for _,x in user_category_score.iterrows()))
    item_features = dataset.build_item_features(((x['game_id'], x['categories']) for _, x in game_df.iterrows()))
    merged_df = pd.merge(user_games_df, game_df[['game_id', 'normalized_score', 'score']], on='game_id', how='inner')

    # Building user -> videogames interaction dataset, using their playtime
    (interactions, weights) = dataset.build_interactions(
        ((x['user_id'], x['game_id'], x['playtime_forever']) for _, x in merged_df.iterrows()))

    model = LightFM()
    epoch_number = 100
    model.fit(interactions=interactions, sample_weight=weights, user_features=users_features,
                  item_features=item_features, epochs=epoch_number, verbose=True)

    user_id = 0  # This is a mock user since the system needs an int to predict the values for given users
    predicted_scores = model.predict(user_ids=user_id, item_ids=np.arange(len(scores)))

    # Exclude games the user has played or owns
    played_or_owned_games = user_games_df.index.tolist()

    # Recommend the top N games with the highest predicted scores
    top_n = 10
    top_game_indices = np.argsort(predicted_scores)[::-1]
    top_recommendations = game_df.loc[top_game_indices]
    top_recommendations['predicted_score'] = predicted_scores[top_game_indices]

    user_game_ids = user_games_df[user_games_df['user_id'] == user_id]['game_id'].tolist()
    top_recommendations = top_recommendations[~top_recommendations['game_id'].isin(user_game_ids)]
    merged_recommendations = pd.merge(top_recommendations, reviews_df[['app_id', 'app_name']], left_on='game_id',
                                      right_on='app_id', how='left')
    merged_recommendations = merged_recommendations.drop_duplicates(subset='game_id')
    merged_recommendations = merged_recommendations[merged_recommendations['normalized_score'] > 9]
    merged_recommendations = merged_recommendations[['app_name', 'predicted_score', 'normalized_score']]
    merged_recommendations = merged_recommendations.head(10)
    merged_recommendations.reset_index(drop=True, inplace=True)