# NYC Taxi Surge Pricing Detection
    The topic of dynamic pricing has made its way into headlines with reasonable backlash. Surge pricing is a form of dynamic pricing aimed at increasing the price of a service, such as transportation, during times of high demand. Companies such as Uber and Lyft employ these tactics during peak travel periods, often costing consumers far more than they expected. Here we will use the New York City (NYC) Taxi and Limousine Commission (TLC) trip record data to predict when surge pricing occurs. As surge pricing is not part of the dataset itself, we will create our own measure of surge pricing to utilize for this analysis. If we can predict when surge pricing occurs, consumers may opt use alternative modes of transportation or change their travel times to avoid excessive pricing.

    The specific datasets used in this study are from the [NYC Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) website, within the time period of February 2019 – February 2026 (the current latest dataset available). Only the High Volume For-Hire Vehicle trips data is being used, a category created specifically for companies that employ drivers and exceed 10,000 trips per day. The datasets contain information about the operating business, dates, times, location, costs, tips, number of passengers, and a few other features. The combined dataset is about 35.6 GB in size (in parquet format) and split into individual files by month with tens of millions of rows in each file. While this may be possible to run on a laptop with libraries such as Dask, it will still take a sizeable amount of time and effort to complete such a task with no reassurance for any failures during runtime. Distributed processing is necessary as it can spread the tasks across multiple computers resulting in faster and efficient processing while also allowing for scalability and fault tolerance.

# 1. Setup and Configuration
     Documentation of the working environment and data dictionary.

## 1.1 SDSC Expanse and Spark Session Builder
    [SDSC Expanse](https://www.sdsc.edu/systems/expanse/) is a high performance computing (HPC) cluster with an impressive system architecture able to handle high-throughput computing and even provide GPU level support. For our setup we will be initializing the cluster with 8 cores (nodes) and 128 GB of memory (per node). 128 GB of memory might seem unwarranted for a data set that is ~35 GB but it is necessary since we will be doing computations, visualizations, and running ML algorithms for this analysis.

    Once logged into SDSC Expanse, we launch a JupyterLab session with the provided cluster information above (8 cores, 128 GB of memory). As this is a shared cluster, there may be a queue before the session becomes available. Open a Jupyter Notebook and initialize a Spark session as seen in the [Spark Session Variables and Build](https://github.com/Autonomousse/nyc_taxi_surge_pricing/blob/master/taxi_surge_pricing.ipynb#Spark-Session-Variables-and-Build) cell. A breakdown of the calculations and corresponding values is below:

    ```
    total_executor_cores = 8 (the total number of cores when initializing the session)
    total_memory = 128 GB (the total memory when initializing the session)
    driver_memory_reserve = 2 GB (the driver coordinates the executors, does not process data)
    
    executor_cores = total_executor_cores - 1 (reserves 1 core for the driver, remaining for executors)
    executor_memory = (total_memory - driver_memory_reserve) / executor_cores

    spark = SparkSession.builder \
    .config("spark.driver.memory", f"{driver_memory_reserve}g") \
    .config("spark.executor.memory", f"{executor_memory}g") \
    .config('spark.executor.instances', executor_cores) \
    .getOrCreate()

    Same as above, but with the calculations and values provided for visual reference:

    spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \     # 2 reserved
    .config("spark.executor.memory", "18g") \  # (128 - 2) / (8 - 1) = 126 / 7 = 18
    .config('spark.executor.instances', 7) \   # 8 - 1 = 7
    .getOrCreate()
    ```
    > [!NOTE]
    > The driver doesn't need much memory as it processes minimal data from aggregations. For visualizations or ML algorithms, may increase to 4 GB.
    
    > [!IMPORTANT]
    > ``` from pyspark.sql import DataFrame, SparkSession # import before initializing spark session builder ```

    
    