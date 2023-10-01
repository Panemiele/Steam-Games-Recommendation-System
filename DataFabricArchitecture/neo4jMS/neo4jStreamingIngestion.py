from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType
from decouple import config


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


def writeInConsole(data_frame):
    return data_frame.format("console").option("truncate", "false").start()


def neo4jStreamingBaseOptions(data_frame):
    return data_frame \
        .format(neo4j_format) \
        .option("url", neo4j_url) \
        .option("save.mode", neo4j_save_mode) \
        .option("checkpointLocation", neo4j_checkpoint_location) \
        .option("authentication.basic.username", config("neo4j_username")) \
        .option("authentication.basic.password", config("neo4j_password"))


def writeStreamingReviewsInNeo4j(data_frame):
    return neo4jStreamingBaseOptions(data_frame) \
        .option("labels", "Review") \
        .option("node.keys", "id") \
        .start()

def writeStreamingGameTagsInNeo4j(data_frame):
    return neo4jStreamingBaseOptions(data_frame) \
        .option("labels", "GameTag") \
        .option("node.keys", "app_id,category") \
        .start()

def writeStreamingUserGameInNeo4j(data_frame):
    return neo4jStreamingBaseOptions(data_frame) \
        .option("labels", "UserGame") \
        .option("node.keys", "user_id,app_id") \
        .start()


def writeStreamingGameInNeo4j(data_frame):
    return neo4jStreamingBaseOptions(data_frame) \
        .option("labels", "Game") \
        .option("node.keys", "app_id") \
        .start()


def writeStreamingGameReviewsRelationshipsInNeo4j(data_frame):
    cypher_query = """
        MATCH (g:Game)
        WITH g
        MATCH (r:Review) 
        WHERE g.app_id = r.app_id
        MERGE (r)-[:IS_WRITTEN_FOR]->(g)
    """
    return neo4jStreamingBaseOptions(data_frame) \
        .option("query", cypher_query) \
        .start()

def writeStreamingGameTagsRelationshipsInNeo4j(data_frame):
    cypher_query = """
        MATCH (g:Game)
        WITH g
        MATCH (t:GameTag) 
        WHERE g.app_id = t.app_id
        MERGE (t)-[:IS_CATEGORY_OF]->(g)
    """
    return neo4jStreamingBaseOptions(data_frame) \
        .option("query", cypher_query) \
        .start()

def writeStreamingGameUsersRelationshipsInNeo4j(data_frame):
    cypher_query = """
        MATCH (g:Game)
        WITH g
        MATCH (u:UserGame) 
        WHERE g.app_id = u.app_id
        WITH distinct u
        MERGE (u)-[:PLAYED]->(g)
    """
    return neo4jStreamingBaseOptions(data_frame) \
        .option("query", cypher_query) \
        .start()



#####################
# VARIABLES SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
application_name = "neo4j-ingestion-util"
kafka_host_port = "localhost:9092"
reviews_topic_name = "reviews-ingestion-topic"
users_topic_name = "users-ingestion-topic"
tags_topic_name = "tags-ingestion-topic"

neo4j_format = "org.neo4j.spark.DataSource"
neo4j_url = "neo4j+s://de233a56.databases.neo4j.io"
neo4j_save_mode = "ErrorIfExists"
neo4j_checkpoint_location = "/tmp/checkpoint/neo4jCheckpoint"

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




################
# MAIN SECTION #
################
# --------------------------------------------------------------------------------------------------#
spark_session = startSparkSession(application_name)
reviews_kafkaStreamReader = readFromKafkaStreamingTopic(kafka_host_port, reviews_topic_name)
tags_kafkaStreamReader = readFromKafkaStreamingTopic(kafka_host_port, tags_topic_name)
users_kafkaStreamReader = readFromKafkaStreamingTopic(kafka_host_port, users_topic_name)

base_reviews_query = reviews_kafkaStreamReader\
    .withColumn("json", from_json(col("value").cast("string"), review_schema)) \
    .withColumn("id", col("json.id")) \
    .withColumn("app_id", col("json.app_id")) \
    .withColumn("app_name", col("json.app_name")) \
    .withColumn("sentiment_target", col("json.sentiment_target")) \
    .withColumn("text", col("json.text")) \
    .withColumn("converted_text", col("json.converted_text"))

reviews_query = base_reviews_query \
    .select("id", "app_id", "app_name", "sentiment_target", "text")\
    .writeStream

games_query = base_reviews_query\
    .select("app_id", "app_name")\
    .distinct()\
    .writeStream

game_tags_query = tags_kafkaStreamReader\
    .withColumn("json", from_json(col("value").cast("string"), tag_schema)) \
    .withColumn("app_id", col("json.app_id")) \
    .withColumn("category", explode(col("json.categories"))) \
    .select("app_id", "category") \
    .writeStream

user_games_query = users_kafkaStreamReader\
    .withColumn("json", from_json(col("value").cast("string"), user_games_schema)) \
    .withColumn("user_id", col("json.user_id")) \
    .withColumn("games", explode(col("json.games"))) \
    .withColumn("app_id", col("games.appid")) \
    .withColumn("playtime_forever", col("games.playtime_forever")) \
    .withColumn("playtime_windows_forever", col("games.playtime_windows_forever")) \
    .withColumn("playtime_mac_forever", col("games.playtime_mac_forever")) \
    .withColumn("playtime_linux_forever", col("games.playtime_linux_forever")) \
    .withColumn("rtime_last_played", col("games.rtime_last_played")) \
    .withColumn("playtime_disconnected", col("games.playtime_disconnected")) \
    .select("user_id", "app_id", "playtime_forever", "playtime_windows_forever", "playtime_mac_forever",
            "playtime_linux_forever", "rtime_last_played", "playtime_disconnected") \
    .writeStream

# console_query = writeInConsole(games_query)
# neo4j_reviews_query = writeStreamingReviewsInNeo4j(reviews_query)
# neo4j_categories_query = writeStreamingGameTagsInNeo4j(game_tags_query)
# neo4j_user_game_query = writeStreamingUserGameInNeo4j(user_games_query)
neo4j_game_query = writeStreamingGameInNeo4j(games_query)
# neo4j_games_reviews_query = writeStreamingGameReviewsRelationshipsInNeo4j(games_query)
# neo4j_games_tags_query = writeStreamingGameTagsRelationshipsInNeo4j(games_query)
# neo4j_games_users_query = writeStreamingGameUsersRelationshipsInNeo4j(games_query)


# spark_session.streams.awaitAnyTermination()