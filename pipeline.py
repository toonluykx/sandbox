from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f
from typing import Tuple

def run_pipeline(spark: SparkSession) -> None:
    airlines, airports, flights = read_sources(spark)

    # 1. Aggregate: average delay per airline / origin / destination airport
    flights_grouped = (
        flights
            .transform(sum_delays)
            .groupBy(["MONTH", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
            .avg("TOTAL_DELAY")
            .withColumnRenamed("avg(TOTAL_DELAY)", "TOTAL_DELAY")
            .select("MONTH", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TOTAL_DELAY")
        ) 

    # 2. Replace airline and airport with full names
    flights_renamed = (
        flights_grouped
            .transform(lambda df: replace_airline(df, airlines))
            .transform(lambda df: replace_airport(df, airports))
            .select("MONTH", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TOTAL_DELAY")
        )
    
    # 3. Split the data up in high/low season df
    low_season, high_season = split_seasons(flights_renamed)

    # 4. Remove the months and group by again
    low_season = (
        low_season
            .drop("MONTH")
            .groupBy(["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
            .avg("TOTAL_DELAY")
            .orderBy(f.desc("avg(TOTAL_DELAY)"))
    )

    high_season = (
        high_season
            .drop("MONTH")
            .groupBy(["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
            .avg("TOTAL_DELAY")
            .orderBy(f.desc("avg(TOTAL_DELAY)"))
    )

    print("======== REPORT FOR LOW SEASON ========")
    low_season.show(10)
    low_season.summary().show()
    low_season.printSchema()
    
    print("======== REPORT FOR HIGH SEASON ========")
    high_season.show(10)
    high_season.summary().show()
    high_season.printSchema()

def sum_delays(flights: DataFrame) -> DataFrame:
    """
    Sum up all different delays into one column "TOTAL_DELAY" and remove the others
    """
    delay_col_names = [col for col in flights.columns if col.endswith("_DELAY")]
    return (
        flights
        .na.fill(0, subset=delay_col_names)
        .withColumn("TOTAL_DELAY", sum([f.col(colname) for colname in delay_col_names]))
        .drop(*delay_col_names)
    )

def print_debug(df: DataFrame) -> None:
    """
    Print header and first row of table
    """
    (
        df
        .select("YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT")
        .show(1)
    )    
    
def replace_airline(df: DataFrame, airlines: DataFrame) -> DataFrame:
    """
    Replace the "AIRLINE" column with the full name of the airline
    """
    return (
        df.
        withColumnRenamed("AIRLINE", "AIRLINE_CODE")
        .crossJoin(airlines)
        .filter(f.col("AIRLINE_CODE") == f.col("IATA_CODE"))
        .drop("AIRLINE_CODE", "IATA_CODE")
    )

def replace_airport(df: DataFrame, airports: DataFrame) -> DataFrame:
    """
    Replace the ORIGIN_AIRPOT and DESTINATION_AIRPORT with the full airport name
    """
    airports1 = (
        airports
        .selectExpr("IATA_CODE as ORIGIN_AIRPORT_CODE", "AIRPORT as ORIGIN_AIRPORT")
    )
    airports2 = (
        airports
        .selectExpr("IATA_CODE as DESTINATION_AIRPORT_CODE", "AIRPORT as DESTINATION_AIRPORT")
    )
    return (
        df
        .withColumnRenamed("ORIGIN_AIRPORT", "ORIGIN_AIRPORT_CODE")
        .withColumnRenamed("DESTINATION_AIRPORT", "DESTINATION_AIRPORT_CODE")
        .join(airports1, on="ORIGIN_AIRPORT_CODE")
        .join(airports2, on="DESTINATION_AIRPORT_CODE")
        .drop("ORIGIN_AIRPORT_CODE", "DESTINATION_AIRPORT_CODE")
    )


def split_seasons(flights: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Split the flights dataset in different parts.
    Returns: 
    - low_season_result (List[DataFrame]): one dataframe of flights per weekday during low season
    - high_season_resul (List[DataFrame]): one dataframe of flights per weekday during high season
    """
    low_season_months = [10, 11, 12, 1, 2, 3]  # Oct -> Mar
    high_season_months = [4, 5, 6, 7, 8, 9]  # Apr -> Sept

    low_season_results = (
        flights
            .filter(f.col("MONTH").isin(low_season_months))
        )

    high_season_results = (
        flights
            .filter(f.col("MONTH").isin(high_season_months))
        )
    
    return low_season_results, high_season_results

def read_sources(spark: SparkSession) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Read inputs and return them as spark dataframes
    """
    airlines = spark.read.csv("data/airlines.csv", header=True, inferSchema=True)
    airports = spark.read.csv("data/airports.csv", header=True, inferSchema=True)
    flights = spark.read.csv("data/flights.csv", header=True, inferSchema=True)

    return airlines, airports, flights
