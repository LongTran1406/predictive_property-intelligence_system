import os
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any
import pandas as pd
import json
import uuid
import logging
import time
from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
sys.path.append(str(project_root / 'app-etl'))
os.chdir(project_root) # Change directory to read the files from ./data folder

from common.utils import read_config

class LoadTool:
    """
    A LoadTool for saving the enriched dataset to cleaned folder
    """
    def __init__(self, config: Dict[str, Any]):
        """
        LoadTool class with a configuration dictionary.
        
        Args:
            config: Dict[str, Any]: Configuration params for path and filename.
        Returns:
            None
        """
        self.config = config
        self.KEYSPACE = "spark_streams"
        self.TABLE = "house_scraping"
        self.CASSANDRA_HOST = "127.0.0.1"
        self.CASSANDRA_PORT = 9042
        self.now = datetime.now()
        self.raw_folder = (
            Path(self.config['data_manager']['raw_data_folder'])
            / f"year={self.now.year}"
            / f"month={self.now.strftime('%m')}"
            / f"day={self.now.strftime('%d')}"
        )
        self.raw_folder.mkdir(parents=True, exist_ok=True)
        self.raw_file = (
            self.raw_folder
            / f"{self.config['data_manager']['raw_database_name'].replace('.parquet','')}_{self.now.strftime('%Y%m%d')}.parquet"
        )

    def gen_uuid(self, x):
        return str(uuid.uuid4()) if x is None else str(x)

    def create_cassandra_connection(self, retries=5, delay=5):
        for i in range(retries):
            try:
                cluster = Cluster([self.CASSANDRA_HOST], port=self.CASSANDRA_PORT)
                session = cluster.connect()
                logging.info("Connected to Cassandra")
                return session
            except Exception as e:
                logging.warning(f"Attempt {i+1}: Cassandra not ready, retrying in {delay}s...")
                time.sleep(delay)
        logging.error("Could not connect to Cassandra after retries")
        return None
    
    def create_keyspace(self, session):
        session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.KEYSPACE}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}};
        """)
        logging.info("Keyspace created successfully!")

    def create_table(self, session):
        session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.{self.TABLE} (
                id UUID primary key,
                address text,
                bathroom_nums text,
                bedroom_nums text,
                car_spaces text,
                land_size text,
                price text,
                lat float,
                lon float,
                postcode text,
                city text
            );
        """)
        logging.info("Table created successfully!")

    def create_spark_connection(self):
        try:
            spark = SparkSession.builder \
                .appName("SparkKafkaToCassandra") \
                .config("spark.jars.packages",
                        "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0,"
                        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2") \
                .config("spark.cassandra.connection.host", self.CASSANDRA_HOST) \
                .config("spark.executor.cores", "1") \
                .config("spark.executor.memory", "512m") \
                .config("spark.cores.max", "1") \
                .getOrCreate()
            spark.sparkContext.setLogLevel("WARN")
            logging.info("Spark session created successfully")
            return spark
        except Exception as e:
            logging.error(f"Could not create Spark session: {e}")
            return None

    def read_kafka_stream(self, spark):
        try:
            df = spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "localhost:9092") \
                .option("subscribe", "house_scraping") \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .load()
            logging.info("Kafka stream connected successfully")
            return df
        except Exception as e:
            logging.error(f"Failed to read Kafka stream: {e}")
            return None
        
    def transform_kafka_df(self, kafka_df):
        schema = StructType([
            StructField("id", StringType(), True),         
            StructField("address", StringType(), True),
            StructField("bathroom_nums", StringType(), True),
            StructField("bedroom_nums", StringType(), True),
            StructField("car_spaces", StringType(), True),
            StructField("land_size", StringType(), True),
            StructField("price", StringType(), True),
            StructField("lat", FloatType(), True),
            StructField("lon", FloatType(), True),
            StructField("postcode", StringType(), True),
            StructField("city", StringType(), True)
        ])

        df = kafka_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        # Convert string ID to Python UUID, then back to string for Spark
        uuid_udf = udf(self.gen_uuid, StringType())
        df = df.withColumn("id", uuid_udf(col("id")))

        return df
    
    def process_batch(self, batch_df, batch_id):
        try:
            count = batch_df.count()  # compute once
            logging.info(f"Processing Batch ID: {batch_id}, Records: {count}")

            if count > 0:
                batch_df.show(5, truncate=False)

                # Write to Cassandra
                batch_df.write \
                    .format("org.apache.spark.sql.cassandra") \
                    .options(table=self.TABLE, keyspace=self.KEYSPACE) \
                    .mode("append") \
                    .save()

                # Append to daily Parquet folder
                batch_df.write \
                    .mode("append") \
                    .parquet(str(self.raw_folder))  # folder path


                logging.info(f"Successfully wrote {count} records to Cassandra and daily Parquet")
            else:
                logging.info("No new records in this batch")

        except Exception as e:
            logging.error(f"✗ Error processing batch {batch_id}: {e}", exc_info=True)
    
    def load(self):
        """
        Function to load enriched dataset to cleaned folder

        Args:
            pd.DataFrame: Input data frame for saving
        Returns:
            None
        """

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        cass_session = self.create_cassandra_connection()
        if cass_session is None:
            exit(1)

        self.create_keyspace(cass_session)
        self.create_table(cass_session)

        spark = self.create_spark_connection()
        if spark is None:
            exit(1)

        kafka_df = self.read_kafka_stream(spark)
        if kafka_df is None:
            exit(1)

        transformed_df = self.transform_kafka_df(kafka_df)
        # transformed_df = transformed_df.repartition(2) 

        # Fresh checkpoint directory
        checkpoint_dir = f"file:///D:/tmp/spark_checkpoint_{int(time.time())}"
        logging.info(f"Using checkpoint: {checkpoint_dir}")

        query = (
            transformed_df.writeStream
            .trigger(processingTime='10 minutes')
            .option("checkpointLocation", checkpoint_dir)
            .foreachBatch(self.process_batch)
            .start()
        )

        logging.info("Streaming started, waiting for data...")
        logging.info("Check Spark UI at http://localhost:4040")
        query.awaitTermination()
        return None
    
if __name__ == "__main__":
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)
    test = LoadTool(config)
    test.load()