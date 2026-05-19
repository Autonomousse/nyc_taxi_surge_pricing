#!/usr/bin/env python
# coding: utf-8

# # Import and Install Dependencies/Packages

# In[1]:


# -----------------------------------------------------------------------------------
# HIDDEN VARIABLES - enviornmental variables
# -----------------------------------------------------------------------------------
#!pip install python-dotenv --user --no-warn-script-location
#from dotenv import load_dotenv
import getpass
user = getpass.getuser()

# -----------------------------------------------------------------------------------
# SPARK SESSION AND OPERATIONS
# -----------------------------------------------------------------------------------
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
#from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# -----------------------------------------------------------------------------------
# DATA AND PYTHON
# -----------------------------------------------------------------------------------

# to capture the data from the URL's where the data is stored
#!pip install requests --user --no-warn-script-location
import os
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
from functools import reduce
import glob
import re
import pandas as pd

# -----------------------------------------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------------------------------------- 
#!pip install plotly --user --no-warn-script-location
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# -----------------------------------------------------------------------------------
# GITIGNORE VARIABLES AND FILES
# -----------------------------------------------------------------------------------

# Username - loaded above with getpass
#load_dotenv() 
#user = os.getenv('user')

# The folder where the data will be stored in the directory.
# Do not upload since it is a lot of data.
folder_name = 'taxi_data'
folder_name_zones = 'taxi_zone'


# # Spark Session Variables and Build

# In[3]:


# -----------------------------------------------------------------------------------
# SET THESE VALUES TO MATCH THE JUPYTER ENVIRONMENT
# -----------------------------------------------------------------------------------

# total cores
#total_executor_cores = 8

# total memory/RAM allocated per node in GB
#total_memory = 128

# set driver memory to 1-2GB for testing and development, 
# up to 4GB for production if necessary
#driver_memory_reserve = 2
'''
# -----------------------------------------------------------------------------------
# THESE VALUES WILL BE CALCULATED AUTOMATICALLY FROM THE VALUES SET ABOVE
# -----------------------------------------------------------------------------------

# reserve 1 core for the driver and rest for executors
executor_cores = total_executor_cores - 1

# amount of memory allocated for each executor
executor_memory = int((total_memory - driver_memory_reserve) / executor_cores)

# -----------------------------------------------------------------------------------
# BUILD THE SPARK SESSION - USE THIS FOR SLURM CLUSTER JOB
# -----------------------------------------------------------------------------------

spark = SparkSession.builder \
    .config("spark.driver.memory", f"{driver_memory_reserve}g") \
    .config("spark.executor.memory", f"{executor_memory}g") \
    .config('spark.executor.instances', executor_cores) \
    .getOrCreate()

'''
# -----------------------------------------------------------------------------------
# BUILD THE SPARK SESSION - THIS IS FOR RUNNING ON JUPYTER LAB ONLY
# -----------------------------------------------------------------------------------

total_executor_cores = 8
total_memory = 128
driver_memory_reserve = 2
overhead_per_executor_gb = 2

executor_cores = total_executor_cores - 1

spark_temp = f'/expanse/lustre/projects/uci157/{user}/nyc_taxi_surge_pricing/spark_temp'

os.makedirs(spark_temp, exist_ok=True)

spark = SparkSession.builder \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.local.dir", spark_temp) \
    .config("spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={spark_temp}") \
    .getOrCreate()



# # Extract, Transform, and Load data into a Spark Dataframe

# In[4]:


# -----------------------------------------------------------------------------------
# IMPORT DATA
# -----------------------------------------------------------------------------------

# Capture the base URL and create a local folder to store the files.
# The base URL will be appended with the dates for the files we want to capture below.
base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_'
base_path = f'/expanse/lustre/projects/uci157/{user}/nyc_taxi_surge_pricing/'
data_folder = f'{base_path}{folder_name}'

# If the folder doesn't exist, create it.
os.makedirs(data_folder, exist_ok=True)

# -----------------------------------------------------------------------------------
# Function to generate the date structure and range for the files we want to download
# Inputs:
# - start_date: datetime format of YYYY-MM-DD
# - end_date: datetime format of YYYY-MM-DD
# -----------------------------------------------------------------------------------
def url_date_gen(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += relativedelta(months=1)

# Define the start and end dates
start_date = datetime(2019,2,1)
end_date = datetime(2026,2,1)

# File counter to track if all files were downloaded already
file_counter = 0

# Download the files that contain the data.
for current_date in url_date_gen(start_date, end_date):
    
    # Format the date to be in the format that the URL uses: YYYY-MM.
    # Generate the URL path with the date.
    file_date = current_date.strftime('%Y-%m')
    file_name = f'High_Volume_Trip_Data_{file_date}.parquet'
    file_url = f'{base_url}{file_date}.parquet'
    file_path = os.path.join(data_folder, file_name)

    # Check if the file exists before attempting download.
    if os.path.exists(file_path):
        file_counter += 1 # add 1 if file is already downloaded
    else:
        # Use try and except to capture any issues.
        # Send a request to download the file.
        try:
            print(f'Downloading... {file_name} from {file_url}...')
            response = requests.get(file_url)
            response.raise_for_status()
    
            # If successful, write the content to a file in parquet.
            # Ensures it is written to disk and not just in memory.
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f'Downloaded {file_name} successfully!')

            # Wait 5 seconds between each call.
            # Prevents denied requests if called to frequently.
            time.sleep(5)
    
        # If there are any issues, print an error message.
        except requests.exceptions.RequestException as e:
            print(f'Failed to download {file_name}: {e}')

# If all files were already downloaded, print a message to let user know.
# Otherwise say how many files were downloaded and skipped.
if file_counter == (end_date - start_date).days + 1:
    print('All files available. No additional files have been downloaded.')
else:
    print(f'{file_counter} files are already available and were skipped.')


# # Download the Taxi Zone Lookup Table

# In[5]:


# Capture the base URL and create a local folder to store the files.
taxi_zone_lookup_url = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv'
taxi_zone_folder = f'{base_path}{folder_name_zones}'

# If the folder doesn't exist, create it.
os.makedirs(taxi_zone_folder, exist_ok=True)

# Set the file path and file name.
file_path = os.path.join(taxi_zone_folder, 'Taxi_Zone_Lookup.csv')

# File counter to track if all files were downloaded already
file_counter = 0

if os.path.exists(file_path):
    file_counter += 1 # add 1 if file is already downloaded
else:
    # Use try and except to capture any issues.
    # Send a request to download the file.
    try:
        print(f'Downloading... Taxi_Zone_Lookup.csv from {taxi_zone_lookup_url}...')
        response = requests.get(taxi_zone_lookup_url)
        response.raise_for_status()

        # If successful, write the content to a file in csv.
        # Ensures it is written to disk and not just in memory.
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded Taxi_Zone_Lookup.csv successfully!')

    # If there are any issues, print an error message.
    except requests.exceptions.RequestException as e:
        print(f'Failed to download Taxi_Zone_Lookup.csv: {e}')

# If all files were already downloaded, print a message to let user know.
# Otherwise say how many files were downloaded and skipped.
if file_counter == 1:
    print('All files available. No additional files have been downloaded.')
else:
    print(f'{file_counter} files are already available and were skipped.')


# #### After attmpting to perform some Spark Dataframe operations, it became obvious that there were data type mismatch issues with the parquet schemas over the years. In order to circumvent any issues with reading the data and performing operations, we will map the appropriate data types across all the files and merge. 
# 
# #### There was also an extra column added into the data beginning in 2025 so we will drop that column for all files that contain it for consistency.

# In[6]:


# -----------------------------------------------------------------------------------
# DEFINE TARGET SCHEMA - the parquet schemas have mismatched types over the years.
# -----------------------------------------------------------------------------------
target_schema = {
    "hvfhs_license_num": StringType(),
    #"dispatching_base_num": StringType(),
    #"originating_base_num": StringType(),
    "request_datetime": TimestampType(),
    #"on_scene_datetime": TimestampType(),
    "pickup_datetime": TimestampType(),
    "dropoff_datetime": TimestampType(),
    "PULocationID": LongType(),
    "DOLocationID": LongType(),
    "trip_miles": DoubleType(),
    "trip_time": LongType(),
    "base_passenger_fare": DoubleType(),
    "tolls": DoubleType(),
    "bcf": DoubleType(),
    "sales_tax": DoubleType(),
    "congestion_surcharge": DoubleType(),
    "airport_fee": DoubleType(),
    "tips": DoubleType(),
    "driver_pay": DoubleType(),
    "shared_request_flag": StringType(),
    "shared_match_flag": StringType()
    #"access_a_ride_flag": StringType(),
    #"wav_request_flag": StringType(),
    #"wav_match_flag": StringType()
}

# There is an extra column starting in 2025, we will drop this for consistency called cbd_congestion_fee.
# We will also remove a few other columns we won't be using.
extra_cols_2025 = ["cbd_congestion_fee", "dispatching_base_num", "originating_base_num", 
                   "on_scene_datetime", "access_a_ride_flag", "wav_request_flag", "wav_match_flag"]

# -----------------------------------------------------------------------------------
# Function to cast and drop extra column, returns a DataFrame
# Inputs:
# - df: is a DataFrame
# - schema_dict: is a dictionary with column names as keys and types as values
# - drop_columns: is a list of column names
# -----------------------------------------------------------------------------------
def cast_and_align(df: DataFrame, schema_dict: dict, drop_columns=None) -> DataFrame:
    # Drop the specific column.
    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in df.columns]
        if cols_to_drop:
            # Unpack all the columns, we only have 1
            df = df.drop(*cols_to_drop)
            # Uncomment print statement if testing for more columns in future.
            # print(f"Dropped columns: {cols_to_drop}")

    # Cast only existing columns, this will resolve any mismatch between the 
    # parquet schemas since there were issues when reading them in originally.
    for col_name, dtype in schema_dict.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(dtype))

    # Reorder columns to match schema, this will ensure unionByName works.
    df = df.select([c for c in schema_dict.keys() if c in df.columns])
    return df

# -----------------------------------------------------------------------------------
# Function to read and process files, returns a DataFrame
# Inputs:
# - folder_path: is the path to the folder as a string
# - schema_dict: is a dictionary with column names as keys and types as values
# - drop_columns: is a list of column names
# -----------------------------------------------------------------------------------
def read_and_process_all_files(folder_path: str, schema_dict: dict) -> DataFrame:
    # Glob is essentially a wildcard search for files, looking for parquet files.
    all_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    dfs = []

    for f in all_files:
        df = spark.read.parquet(f) # read files one at a time

        # Drop extra columns only for files that are after the year 2024.
        # More years can be added here if processing at a later date.
        match = re.search(r'\d{4}', f) # finds year value 2024
        year = int(match.group()) if match else None
        if year and year > 2024:
            df = cast_and_align(df, schema_dict, drop_columns=extra_cols_2025)
        else:
            df = cast_and_align(df, schema_dict)

        dfs.append(df)

    # If there are no valid files in the folder, throw an exception.
    if not dfs:
        raise ValueError("No parquet files found in the folder...")

    # Merge all DataFrames.
    df_all = reduce(DataFrame.unionByName, dfs)
    return df_all

# -----------------------------------------------------------------------------------
# LOAD AND MERGE ALL DATA
# -----------------------------------------------------------------------------------
# folder_name was set previously, it is where the data was downloaded.
df = read_and_process_all_files(data_folder, target_schema)

# -----------------------------------------------------------------------------------
# VALIDATE BY CHECKING SCHEMA
# -----------------------------------------------------------------------------------
#df.printSchema()


# #### To see the driver and executors that are running

# In[7]:


# Run this cell after initializing SparkSession and loading in data.

# Get the active Spark Context and URL
try:
    sc = spark.sparkContext
    url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/executors"
    
    # Fetch the executor data from the API
    response = requests.get(url)
    executors = response.json()
    
    # Format into a readable DataFrame
    df_exec = pd.DataFrame(executors)[['id', 'totalCores', 'maxMemory', 'activeTasks', 'isActive']]
    df_exec['maxMemory_GB'] = (df_exec['maxMemory'] / (1024**3)).round(2)
    print(df_exec)
except Exception as e:
    print(f"Could not fetch executor info: {e}")

# removing this section since it's already done in the notebook, saves processing time.
'''


# # Number of Observations

# In[8]:

# removing to save processing since already captured in notebook 
total_rows = df.count()
print('Total observations: {:,}'.format(total_rows))


# # Check for Duplicates

# In[9]:


# We will use pickup_datetime to see if there are duplicates.
# Since using distinct() would require a shuffle and we don't have distinct row identifiers.
column_name = "pickup_datetime"
column_df = df.select(column_name)

# Group by the column and count occurrences.
duplicate_check_df = column_df.groupBy(column_name).count()

# Filter for duplicates (count > 1).
duplicates_df = duplicate_check_df.filter(F.col("count") > 1)

# Sort by count in descending order.
sorted_duplicates_df = duplicates_df.orderBy(F.col("count").desc())

# Show duplicates.
sorted_duplicates_df.show()


# # Quick Stats on Numerical Columns

# In[10]:


df.describe('trip_miles', 'trip_time', 'base_passenger_fare', 'tolls', 'bcf', 
                'sales_tax', 'congestion_surcharge', 'airport_fee', 'tips', 'driver_pay').show(vertical=True)


# # Exploratory Data Analysis and Visualizations

# ## Surge Conditions - When are surge conditions more likely?

# In[11]:


# Create a new column for requested pickup hour and pickup day of the week.
df = df.withColumn("request_hour", F.hour("request_datetime"))
df = df.withColumn("pickup_day_of_week", F.dayofweek("request_datetime")) # dayofweek starts with Sunday as 1

# Group by pickup hour and pickup day of the week to get counts.
hourly_demand = df.groupBy("request_hour").count().orderBy("request_hour")
day_of_week_demand = df.groupBy("pickup_day_of_week").count().orderBy("pickup_day_of_week")

# Create a folder to save the visualizations in.
image_folder = "visualizations"
os.makedirs(image_folder, exist_ok=True)

# Convert to a Pandas DF to plot a histogram.
hourly_pd_df = hourly_demand.toPandas()
plt.plot(hourly_pd_df['request_hour'], hourly_pd_df['count'])
plt.title("Trips per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")

# Save the chart.
plt.savefig(f"{image_folder}/trips_per_hour.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close()


# In[12]:


# Convert to a Pandas DF to plot a bar chart.
day_of_week_pd_df = day_of_week_demand.toPandas()

# Map the values to the day
day_names = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
day_of_week_pd_df['day_name'] = day_of_week_pd_df['pickup_day_of_week'].map(day_names)#.astype(str)

# Drop rows where mapping failed (NaN)
day_of_week_pd_df = day_of_week_pd_df.dropna(subset=['day_name'])

# Define the output order
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Reorder the DataFrame accordingly
day_of_week_pd_df['day_name'] = pd.Categorical(day_of_week_pd_df['day_name'], categories=day_order, ordered=True)
day_of_week_pd_df = day_of_week_pd_df.sort_values('day_name')

#plt.figure(figsize=(10,6))
plt.bar(day_of_week_pd_df['day_name'], day_of_week_pd_df['count'])
plt.title("Trips per Day of Week")
plt.xlabel("Day of Week")
plt.xticks(rotation=30)
plt.ylabel("Number of Trips")

# Save the chart.
plt.savefig(f"{image_folder}/trips_per_day_of_week.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close()


# ## Price Vs Distance - Are there unusual prices?

# In[13]:


# Remove: trip time below 0 and above 7200 (2 hours), fare under 0, and cap miles at 100.
df_filtered_time = df.filter((F.col("trip_time") > 0) & (F.col("trip_time") <= 7200) & (F.col("base_passenger_fare") > 0) & (F.col("trip_miles") < 100))

# Take a sample of the data so we don't plot 1.5 billion points.
df_sample_time = df_filtered_time.sample(fraction=0.001, seed=24)

# Convert to Pandas for plotting
df_pd_time = df_sample_time.select("trip_time", "base_passenger_fare").toPandas()

plt.figure(figsize=(10,6))
plt.scatter(df_pd_time['trip_time'], df_pd_time['base_passenger_fare'], alpha=0.3, s=5)
plt.xlabel("Trip Time (seconds)")
plt.ylabel("Base Passenger Fare ($)")
plt.title("Fare vs Trip Time")
plt.grid(True)

# Save the chart.
plt.savefig(f"{image_folder}/fare_vs_trip_time.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close()


# In[14]:


# Remove: trip miles under 0, fare under 0, and cap miles at 100.
df_filtered = df.filter((F.col("trip_miles") > 0) & (F.col("base_passenger_fare") > 0) & (F.col("trip_miles") < 100))

# Take a sample of the data so we don't plot 1.5 billion points.
df_sample = df_filtered.sample(fraction=0.001, seed=24)

# Convert to Pandas for plotting
df_pd = df_sample.select("trip_miles", "base_passenger_fare").toPandas()

plt.figure(figsize=(10,6))
plt.scatter(df_pd['trip_miles'], df_pd['base_passenger_fare'], alpha=0.3, s=5)
plt.xlabel("Trip Miles")
plt.ylabel("Base Passenger Fare ($)")
plt.title("Fare vs Distance")
plt.grid(True)

# Save the chart.
plt.savefig(f"{image_folder}/fare_vs_distance.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close()


# ## Demand Vs Price - Is there a relationship here?

# In[15]:


# Get the taxi zone lookup csv for mapping the zone names.
zone_lookup_df = pd.read_csv(f"{taxi_zone_folder}/Taxi_Zone_Lookup.csv")

# Remove: trip miles under 0, fare under 0, and cap miles at 100.
df_filtered_price = df.filter((F.col("trip_miles") > 0) & (F.col("base_passenger_fare") > 0) & (F.col("trip_miles") < 100))

# Take a sample of the data so we don't plot 1.5 billion points.
df_sample_price = df_filtered_price.sample(fraction=0.001, seed=24)

# Compute fare per mile.
df_price = df_sample_price.withColumn("fare_per_mile", F.col("base_passenger_fare") / F.col("trip_miles"))

# Sample with lower average.
df_price = df_price.filter((F.col("fare_per_mile") > 0) & (F.col("fare_per_mile") <= 50))

# Extract hour from pickup.
df_price = df_price.withColumn("pickup_hour", F.hour("pickup_datetime"))

# Count trips per zone per hour
demand_hour_zone = df_price.groupBy("PULocationID", "pickup_hour").count().withColumnRenamed("count", "trips")

# Average fare per mile per zone per hour
avg_fare_hour_zone = df_price.groupBy("PULocationID", "pickup_hour").agg(F.avg("fare_per_mile").alias("avg_fare_per_mile"))

# Join the trip counts with the average far per mile.
combined = demand_hour_zone.join(avg_fare_hour_zone, ["PULocationID", "pickup_hour"])

# Convert Spark DF to Pandas DF for plotting.
combined_pd_df = combined.toPandas()

# Map the names of the zones to make them more legible.
combined_pd_df = combined_pd_df.merge(zone_lookup_df, left_on="PULocationID", right_on="LocationID", how="left")


plt.figure(figsize=(10,6))
plt.scatter(combined_pd_df['trips'], combined_pd_df['avg_fare_per_mile'], alpha=0.3)
plt.xlabel("Trips per Hour in Zone")
plt.ylabel("Average Fare per Mile ($)")
plt.title("Trips vs Average Fare per Mile")

# Save the chart.
plt.savefig(f"{image_folder}/trips_vs_avg_fare_per_mile.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close()


# ### Unique Employer Breakdown - Is there an even distribution or does one company have a majority?

# In[16]:


# Count trips per employer.
driver_counts = df.groupBy("hvfhs_license_num").agg(F.count("*").alias("num_trips"))

# Collect to Pandas DF for plotting.
driver_counts_pd_df = driver_counts.toPandas()

# Add the names from the data dictionary.
driver_counts_pd_df['business_name'] = driver_counts_pd_df['hvfhs_license_num'].map({
    "HV0002": "Juno",
    "HV0003": "Uber",
    "HV0004": "Via",
    "HV0005": "Lyft",
}).fillna("Unknown")

# Organize in ascending value.
driver_counts_pd_df = driver_counts_pd_df.sort_values('num_trips', ascending=True)

# Calculate the percentage of total trips to display.
total_trips = driver_counts_pd_df['num_trips'].sum()
driver_counts_pd_df['pct'] = driver_counts_pd_df['num_trips'] / total_trips * 100


plt.figure(figsize=(8,5))
bars = plt.barh(driver_counts_pd_df['business_name'], driver_counts_pd_df['num_trips'])

# Add some space to the x-axis so the percentage doesn't get cut off.
plt.xlim(0, driver_counts_pd_df['num_trips'].max() * 1.1)
plt.xlabel("Number of Trips")
plt.title("Trips per Employer")

# Add percentages to each bar.
for bar, pct in zip(bars, driver_counts_pd_df['pct']):
    width = bar.get_width()
    plt.text(width + total_trips*0.005, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%", va='center')

plt.tight_layout()

# Save the chart.
plt.savefig(f"{image_folder}/trips_per_employer.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close()


# # Data Cleaning and Preprocessing

# In[17]:


# We saw in the quick stats above that there are values that don't make sense.
# First we will clean up the df by removing rows that seem to be outside of normal operation.

# By checking the 99th percentile of each of these values, we can find an upper bound to cap off the data.
# For example, trip_miles for most trips, based on the mean and standard deviation, is most likely not over 100 miles (most likely less).
df.approxQuantile(
    ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay"],
    [0.001, 0.01, 0.99, 0.999, 0.9999],
    0.001
)


'''


# The values for the 99th percentile indicate that we can use the following caps:
# - trip_miles: 30 miles
# - trip_time: 4500 seconds
# - passenger_base_fare: 110 dollars
# - driver_pay: 80 dollars
# 
# This gives us a bit of a buffer between the 99th percentile values and the cap. The 99.9th and 99.99th percentiles also are the same as the max values from the quick stats which means these might be erroneous values and should be removed anyways.

# In[18]:


df_filtered = df.filter(
    (F.col("trip_miles") > 0.5) &            # exclude near zero/extremely short distances
    (F.col("trip_miles") <= 30) &            # exclude implausible long trips
    (F.col("trip_time") >= 180) &            # exclude trips less than 3 minutes
    (F.col("trip_time") <= 4500) &           # exclude trips over 1 hour and 15 minutes (4500 seconds)
    (F.col("base_passenger_fare") > 3) &     # exclude zero and negative fares
    (F.col("base_passenger_fare") <= 110) &  # exclude fares over 110 dollars
    (F.col("driver_pay") > 0) &              # exclude zero and negative driver pay
    (F.col("driver_pay") <= 80)              # exclude driver pay over 80 dollars
)


# In[19]:

'''

# Checking to see if the values are more reasonable
df_filtered.describe('trip_miles', 'trip_time', 'base_passenger_fare', 'tips', 'driver_pay').show(vertical=True)


# Since we are going turn tolls, congestion_surcharge, and airport_fee into binary columns, the max values don't really matter here. Otherwise the maxes here do make sense but might need to be checked. sales_tax and bcf will be dropped so the values don't matter much here but good to check just to be safe. tips max value of 500 seems unreasonable for an average of ~$22 base fare so we will check the percentiles to see what is happening here.

# In[20]:


df_filtered.approxQuantile(
    "tips",
    [0.95, 0.99, 0.999],
    0.001
)


# In[21]:

'''

# We can set a cap at 20 for tips since we see a similar pattern of the values jumping to the max value as before.
df_filtered = df_filtered.filter(F.col("tips") <= 20)


# In[22]:


# Also want to make sure this is for single riders and not for shared rides because shared rides are typically discounted.
# Remove the shared rides.
df_filtered = df_filtered.filter(F.col("shared_match_flag") != 'Y')

#df_filtered.count()


# # Feature Engineering

# In[23]:


# Drop any columns that we created earlier for charts or exploratory analysis as our definitions may change for feature engineering
df_filtered = df_filtered.drop("request_hour", "pickup_day_of_week")


# ### Price Per Mile Ratio VS Median Baseline - Target

# In[24]:


# First create the features we will use to calculate the rate
# 1 = Sunday, 7 = Saturday
# F.lit(None) fills in values less than 0 with null
df_filtered = df_filtered.withColumn("hour_of_day", F.hour("request_datetime")) \
                         .withColumn("day_of_week", F.dayofweek("request_datetime")) \
                         .withColumn("month", F.month("request_datetime")) \
                         .withColumn("is_weekend", (F.dayofweek("request_datetime").isin([1,7])).cast("int")) \
                         .withColumn("wait_time_secs", F.unix_timestamp("pickup_datetime") - F.unix_timestamp("request_datetime")) \
                         .withColumn("wait_time_secs", F.when(F.col("wait_time_secs") < 0, F.lit(None)).otherwise(F.col("wait_time_secs"))) \
                         .withColumn("fare_per_mile", F.col("base_passenger_fare") / F.col("trip_miles")) \
                         .withColumn("fare_per_min", F.col("base_passenger_fare") / F.col("trip_time") / 60)


# In[25]:

'''

# Check to see if fare_per_mile is valid
df_filtered.describe('fare_per_mile').show(vertical=True)


# In[26]:


# There seems to be a very small fare_per_mile for the min value.
# We want to filter these out because these are unreasonable and most likely errors in the data.
df_filtered.approxQuantile(
    "fare_per_mile",
    [0.001, 0.01, 0.99, 0.999],
    0.001
)


# In[27]:

'''


# Set the min amount to $2/mile and the max to $25/mile
df_filtered = df_filtered.filter((F.col("fare_per_mile") >= 2.0) &(F.col("fare_per_mile") <= 25.0))

# Persist here and materialize to improve performance by caching the data on disk
df_filtered.persist(StorageLevel.DISK_ONLY)
df_filtered.count()


# In[28]:

'''
# Check to see if base_passenger_fare is valid
df_filtered.describe('base_passenger_fare').show(vertical=True)
'''

# In[29]:


# Calculate the baseline median per group as an aggregation: platform + pickup zone + hour + day of week
#baseline_window = Window.partitionBy("hvfhs_license_num", "PULocationID", "hour_of_day", "day_of_week")
#df_filtered = df_filtered.withColumn("median_fare_per_mile", F.percentile_approx("fare_per_mile", 0.5).over(baseline_window))
median_df = df_filtered.groupBy("hvfhs_license_num", "PULocationID", "hour_of_day", "day_of_week") \
    .agg(F.percentile_approx("fare_per_mile", 0.5).alias("median_fare_per_mile"))

# Join back to df
df_filtered = df_filtered.join(
    median_df,
    on=["hvfhs_license_num", "PULocationID", "hour_of_day", "day_of_week"],
    how="left"
)

# Compute the surge multiplier and create target label
df_filtered = df_filtered.withColumn("surge_multiplier", F.col("fare_per_mile") / F.col("median_fare_per_mile"))

# The 75th percentile of surge_multiplier gives us a consistent ~25% surge label class regardless of overall rate levels.
#surge_threshold = df_filtered.approxQuantile("surge_multiplier", 0.75, 0.001)[0] --------------------------------------- can test to see if this works better
surge_threshold = 1.25 # 25% over median baseline means surge condition

df_filtered = df_filtered.withColumn("is_surge", (F.col("surge_multiplier") >= surge_threshold).cast("int"))


# ### Demand Density - Surge occurs when demand outpaces supply

# In[30]:


# Approximate demand by counting trips per zone within a rolling time window and comparing it to the historical trend.

# Bucket time into 15 minute intervals.
df_filtered = df_filtered.withColumn("time_bucket", (F.unix_timestamp("request_datetime") / 900).cast("long"))  # 900s = 15 min


# Count trips per zone per 15 min bucket.
demand_counts = df_filtered.groupBy("PULocationID", "time_bucket").agg(F.count("*").alias("trips_in_window"))

# Compute historical average and std for zone + hour + day of week
demand_counts = demand_counts.withColumn("hour_of_day", ((F.col("time_bucket") * 900 / 3600) % 24).cast("int")) \
                             .withColumn("day_of_week", F.dayofweek(F.from_unixtime(F.col("time_bucket") * 900)))

demand_baseline = demand_counts.groupBy("PULocationID", "hour_of_day", "day_of_week").agg(
        F.mean("trips_in_window").alias("avg_demand"),
        F.stddev("trips_in_window").alias("std_demand"))

demand_counts = demand_counts.join(demand_baseline, on=["PULocationID", "hour_of_day", "day_of_week"], how="left")

demand_counts = demand_counts.withColumn("demand_zscore", (F.col("trips_in_window") - F.col("avg_demand")) / F.col("std_demand")) \
                             .withColumn("high_demand", (F.col("demand_zscore") >= 1.5).cast("int"))  # 1.5 std devs above average

# Join back to main df, df_filtered
df_filtered = df_filtered.join(demand_counts.select("PULocationID", "time_bucket", "high_demand", "demand_zscore"),
                               on=["PULocationID", "time_bucket"],
                               how="left")


# ### Combined Target - is_surge + high_demand

# In[31]:


# Combining both surge and demand signals would strengthen our confidence that surge pricing is occurring.
# df_filtered = df_filtered.withColumn("is_surge_combined", ((F.col("surge_multiplier") >= 1.25) & (F.col("high_demand") == 1)).cast("int"))

# can test this instead of is_surge, if we get similar or better results we can swap since this would be a better signal


# ### Driver Pay Ratio - Cross check with passenger fare. Both increasing can validate surge pricing

# In[32]:


df_filtered = df_filtered.withColumn("driver_pay_ratio", F.col("driver_pay") / F.col("base_passenger_fare"))

# Filter any driver pay ratio over 1 because this means the driver was paid more than the base fare. Possible with incentives or multipliers.
# Or it could just be corrupt data. We can allow a ratio of 2, which means the driver can earn up to 2x base fare with any incentives
# before we consider it as possible outliers.
df_filtered = df_filtered.withColumn("driver_pay_ratio", F.when(F.col("driver_pay") / F.col("base_passenger_fare") > 2.0, F.lit(2.0)) \
                                     .otherwise(F.col("driver_pay") / F.col("base_passenger_fare")))


# ### Mapping hvfhs_license_num to integer

# In[33]:


# Want to map to integer from string for ML
indexer = StringIndexer(inputCol="hvfhs_license_num", outputCol="license_index")
df_filtered = indexer.fit(df_filtered).transform(df_filtered)


# ### Mapping tolls, airport_fee, and congestion_surcharge to binary

# In[34]:


df_filtered = df_filtered.withColumn("has_toll", (F.col("tolls") > 0).cast("int")) \
                         .withColumn("has_airport_fee", (F.col("airport_fee") > 0).cast("int")) \
                         .withColumn("has_congestion_surcharge", (F.col("congestion_surcharge") > 0).cast("int"))


# ### Borough Indices - Location ID's are good for fine grained geographic signals but boroughs might give segmented signals

# In[35]:


# Read the lookup table
zone_lookup = spark.read.csv(f"{taxi_zone_folder}/Taxi_Zone_Lookup.csv", header=True)

# Pickup borough
df_filtered = df_filtered.join(zone_lookup.select(
        F.col("LocationID").cast("int"),
        F.col("Borough").alias("PU_Borough")),
        df_filtered.PULocationID == F.col("LocationID"), how="left").drop("LocationID")

# Dropoff borough
df_filtered = df_filtered.join(zone_lookup.select(
        F.col("LocationID").cast("int"),
        F.col("Borough").alias("DO_Borough")),
        df_filtered.DOLocationID == F.col("LocationID"), how="left").drop("LocationID")

# Encode borough with StringIndexer (6 unique values)
for col_name, out_name in [("PU_Borough", "PU_borough_index"), ("DO_Borough", "DO_borough_index")]:
    borough_indexer = StringIndexer(inputCol=col_name, outputCol=out_name, handleInvalid="keep")
    df_filtered = borough_indexer.fit(df_filtered).transform(df_filtered).drop(col_name)


# In[36]:


# Drop columns that are no longer required
df_model = df_filtered.drop(
    # Raw categoricals replaced by encoded versions
    "hvfhs_license_num",
    # Raw datetimes replaced by extracted features
    "request_datetime", "pickup_datetime", "dropoff_datetime",
    # Raw fee columns replaced by binary versions
    "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee",
    # Filtered out flags
    "shared_request_flag", "shared_match_flag",
    # Intermediate computation columns
    "time_bucket", "median_fare_per_mile",
    # Direct label leakage since this would predict >= 1.25 = target label
    "surge_multiplier"
)

# Persist here to improve performance by caching the data on disk
df_model.persist(StorageLevel.DISK_ONLY)


# In[ ]:

'''

# Checking for class imbalance, we want there to be enough surge labels so that we have a balanced data set
counts = df_model.groupBy("is_surge").count().collect()

total_rows = sum(row["count"] for row in counts)
for row in counts:
    print(f"is_surge={row['is_surge']}: {row['count']:,} ({round(row['count']/total_rows*100, 2)}%)")

'''

# Filtering out nulls for is_surge
df_model = df_model.filter(F.col("is_surge").isNotNull())

# # Model Setup - Training, Testing, Validation

# In[ ]:


# Get counts dynamically
counts = df_model.groupBy("is_surge").count().collect()
total_rows = sum(row["count"] for row in counts)
count_dict = {row["is_surge"]: row["count"] for row in counts}

# Calculate weights inversely proportional to class frequency
surge_count = count_dict[1]
non_surge_count = count_dict[0]

# Print class distribution
print(f"Total rows: {total_rows:,}")
for row in counts:
    print(f"is_surge={row['is_surge']}: {row['count']:,} ({round(row['count']/total_rows*100, 2)}%)")

# Calculate weights automatically; weight = total / (num_classes * class_count)
weight_surge = total_rows / (2 * surge_count)
weight_non_surge = total_rows / (2 * non_surge_count)

print(f"Surge weight: {weight_surge:.4f}")
print(f"Non-surge weight: {weight_non_surge:.4f}")

# Add weight column
df_model = df_model.withColumn(
    "class_weight",
    F.when(F.col("is_surge") == 1, weight_surge).otherwise(weight_non_surge)
)

# Set up the features using VectorAssembler
feature_cols = [
    "PULocationID", "DOLocationID", "PU_borough_index", "DO_borough_index",
    "license_index", "trip_miles", "trip_time", "base_passenger_fare",
    "tips", "driver_pay", "fare_per_mile", "fare_per_min", "driver_pay_ratio",
    "has_toll", "has_airport_fee", "has_congestion_surcharge",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "wait_time_secs", "demand_zscore", "high_demand"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df_assembled = assembler.transform(df_model)

# Train, test, split
train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Persist both splits so they don't get recomputed for evaluation
train_df.persist(StorageLevel.DISK_ONLY)
test_df.persist(StorageLevel.DISK_ONLY)
train_df.count()
test_df.count()

# Train the model
dt = DecisionTreeClassifier(
    labelCol="is_surge",
    featuresCol="features",
    weightCol="class_weight",
    maxDepth=10,          # tune this, default is 5
    maxBins=265,          # may need to tune this, 512 to safely handle all categorical features like location ID
    seed=42
)

dt_model = dt.fit(train_df)


# Model evaluation
predictions = dt_model.transform(test_df)

# AUC-ROC
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="is_surge",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = binary_evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Accuracy, F1, Precision, Recall
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="is_surge",
    predictionCol="prediction"
)

for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    score = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: metric})
    print(f"{metric}: {score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": dt_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)
print(feature_importance.to_string())