# NYC Taxi Surge Pricing Detection
The topic of dynamic pricing has made its way into headlines with reasonable backlash. Surge pricing is a form of dynamic pricing aimed at increasing the price of a service, such as transportation, during times of high demand. Companies such as Uber and Lyft employ these tactics during peak travel periods, often costing consumers far more than they expected. Here we will use the New York City (NYC) Taxi and Limousine Commission (TLC) trip record data to predict when surge pricing occurs. As surge pricing is not part of the dataset itself, we will create our own measure of surge pricing to utilize for this analysis. If we can predict when surge pricing occurs, consumers may opt use alternative modes of transportation or change their travel times to avoid excessive pricing.

The specific datasets used in this study are from the [NYC Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) website, within the time period of February 2019 – February 2026 (the current latest dataset available). Only the High Volume For-Hire Vehicle trips data is being used, a category created specifically for companies that employ drivers and exceed 10,000 trips per day. The datasets contain information about the operating business, dates, times, location, costs, tips, number of passengers, and a few other features. The combined dataset is about 35.6 GB in size (in parquet format) and split into individual files by month with tens of millions of rows in each file. While this may be possible to run on a laptop with libraries such as Dask, it will still take a sizeable amount of time and effort to complete such a task with no reassurance for any failures during runtime. Distributed processing is necessary as it can spread the tasks across multiple computers resulting in faster and efficient processing while also allowing for scalability and fault tolerance.

# 1. Setup and Configuration
Documentation of the working environment and data dictionary.

## 1.1 File Structure
Within the SDSC Environment, the file structure is denoted below:

```
/home/<username>/expanse/lustre/projects/uci157/<username>/nyc_taxi_surge_pricing/
|--- taxi_data/                        # parquet data files - not uploaded to Github
|--- taxi_zone/                        # CSV file containing zone lookup values - not uploaded to Github
|--- visualizations/                   # visualizations generated for analysis
|--- taxi_surge_pricing.ipynb          # notebook with code for analysis
```
> [!TIP]
> The data and visualization folders will be created automatically if you run the Jupyter Notebook.

## 1.2 SDSC Expanse and Spark Session Builder
[SDSC Expanse](https://www.sdsc.edu/systems/expanse/) is a high performance computing (HPC) cluster with an impressive system architecture able to handle high-throughput computing and even provide GPU level support. For our setup we will be initializing the cluster with 8 cores (nodes) and 128 GB of memory (per node). 128 GB of memory might seem unwarranted for a data set that is ~35 GB but it is necessary since we will be doing computations, visualizations, and running ML algorithms for this analysis.

Once logged into SDSC Expanse, we launch a JupyterLab session with the provided cluster information above (8 cores, 128 GB of memory). As this is a shared cluster, there may be a queue before the session becomes available. After it launches, open a Jupyter Notebook and initialize a Spark session as seen in the [Spark Session Variables and Build](https://github.com/Autonomousse/nyc_taxi_surge_pricing/blob/master/taxi_surge_pricing.ipynb#Spark-Session-Variables-and-Build) cell. A breakdown of the calculations and corresponding values are below:

```python
total_executor_cores = 8      # the total number of cores when initializing the session
total_memory = 128 GB         # the total memory when initializing the session
driver_memory_reserve = 2 GB  # the driver coordinates the executors, does not process data

executor_cores = total_executor_cores - 1 (reserves 1 core for the driver, remaining for executors)
executor_memory = (total_memory - driver_memory_reserve) / executor_cores

spark = SparkSession.builder \
.config("spark.driver.memory", f"{driver_memory_reserve}g") \
.config("spark.executor.memory", f"{executor_memory}g") \
.config('spark.executor.instances', executor_cores) \
.getOrCreate()

# Same as above, but with the calculations and values provided for visual reference:
spark = SparkSession.builder \

# 2 reserved for the driver
.config("spark.driver.memory", "2g") \

# (128 - 2) / (8 - 1) = 126 / 7 = 18
.config("spark.executor.memory", "18g") \

# 8 - 1 = 7
.config('spark.executor.instances', 7) \
.getOrCreate()
```
> [!NOTE]
> The driver doesn't need much memory as it processes minimal data from aggregations. For visualizations or ML algorithms, may increase to 4 GB.

> [!IMPORTANT]
> ```from pyspark.sql import SparkSession # import before initializing spark session builder```
> If you are using the provided notebook, this has already been done at the top.

A screenshot of the driver and total memory after loading in the data:
![spark_screenshot](visualizations/spark_screenshot.PNG)

## 1.3 Running the Jupyter Notebook
To run the notebook, the following criteria must be met (or workarounds must be created by the user):

1. Create a .env file in the same directory as the notebook and enter the following:
    - ```user="<username>" # <username> is your username on the server.```
    - This will be read into the notebook automatically to set the path for folder creation.
    - Helps to keep your information safe if you intend to upload your work online.
2. Clone the repo into your user location: ../username/***clone-here***
3. In the Jupter Notebook, under the section labeled [Extract, Transform, and Load data into a Spark Dataframe](https://github.com/Autonomousse/nyc_taxi_surge_pricing/blob/master/taxi_surge_pricing.ipynb#Extract,-Transform,-and-Load-data-into-a-Spark-Dataframe) adjust the file location for the ***base_path*** prior to the username:
    - ```base_path = f'<<<set your folder path up to your username here>>>/{user}/nyc_taxi_surge_pricing/'```
> [!WARNING]
> Please review the dependencies at the top of the notebook prior to running. Some less common dependencies have been added as an install, more may be needed depending on your environment.

# 2. Data Source and Data Dictionary

## 2.1 Data Source
The entire data set can be found here: [NYC Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- The specific files we are using are the **High Volume For-Hire Vehicle Trip Records** starting from February 2019 to February 2026 (inclusive). These files are available in parquet format, which is the preferred format for analyzing large sets of data.
- The Taxi Zone Maps and Lookup Table in CSV format will also be utilized for this analysis.
- A data dictionary is also provided on the website and here.
> [!TIP]
> If using the provided notebook, all of the files will be downloaded automatically.
  
## 2.2 Data Dictionary

| Field Name                 | Description                                                                                          |
|----------------------------|------------------------------------------------------------------------------------------------------|
| **hvfhs_license_num**      | The TLC license number of the HVFHS base or business.                                                |
| **dispatching_base_num**   | The TLC Base License Number of the base that dispatched the trip.                                    |
| **originating_base_num**   | Base number of the base that received the original trip request.                                     |
| **request_datetime**       | Date/time when passenger requested to be picked up.                                                  |
| **on_scene_datetime**      | Date/time when driver arrived at the pick-up location (Accessible Vehicles-only).                    |
| **pickup_datetime**        | The date and time of the trip pick-up.                                                               |
| **dropoff_datetime**       | The date and time of the trip drop-off.                                                              |
| **PULocationID**           | TLC Taxi Zone in which the trip began.                                                               |
| **DOLocationID**           | TLC Taxi Zone in which the trip ended.                                                               |
| **trip_miles**             | Total miles for passenger trip.                                                                      |
| **trip_time**              | Total time in seconds for passenger trip.                                                            |
| **base_passenger_fare**    | Base passenger fare before tolls, tips, taxes, and fees.                                             |
| **tolls**                  | Total amount of all tolls paid in trip.                                                              |
| **bcf**                    | Total amount collected in trip for Black Car Fund.                                                   |
| **sales_tax**              | Total amount collected in trip for NYS sales tax.                                                    |
| **congestion_surcharge**   | Total amount collected in trip for NYS congestion surcharge.                                         |
| **airport_fee**            | $2.50 for both drop off and pick up at LaGuardia, Newark, and John F. Kennedy airports.              |
| **tips**                   | Total amount of tips received from passenger.                                                        |
| **driver_pay**             | Total driver pay (not including tolls or tips and net of commission, surcharges, or taxes).          |
| **shared_request_flag**    | Did the passenger agree to a shared/pooled ride, regardless of whether they were matched? (Y/N)      |
| **shared_match_flag**      | Did the passenger share the vehicle with another passenger who booked separately at any point? (Y/N) |
| **access_a_ride_flag**     | Was the trip administered on behalf of the Metropolitan Transportation Authority (MTA)? (Y/N)        |
| **wav_request_flag**       | Did the passenger request a wheelchair-accessible vehicle (WAV)? (Y/N)                               |
| **wav_match_flag**         | Did the trip occur in a wheelchair-accessible vehicle (WAV)? (Y/N)                                   |
| **cbd_congestion_fee**     | Per-trip charge for MTA's Congestion Relief Zone starting Jan. 5, 2025.                              |

## 2.3 Distribution of the Data

These are the columns that we will be using for our analysis:

| Field Name                 | Description                                                                                    |
|----------------------------|------------------------------------------------------------------------------------------------|
| **hvfhs_license_num**      | 4 unique categorical identifiers for Juno (HV0002), Uber(HV0003), Via (HV0004), Lyft (HV0005). |
| **request_datetime**       | Continuous datetime variable for requested pickup time.                                        |
| **pickup_datetime**        | Continuous datetime for pick-up.                                                               |
| **dropoff_datetime**       | Continuous datetime for drop-up.                                                               |
| **PULocationID**           | Categorical zone identifier where the trip began. 265 values.                                  |
| **DOLocationID**           | Categorical zone identifier where the trip ended. 265 values.                                  |
| **trip_miles**             | Continuous variable for total miles.                                                           |
| **trip_time**              | Continuous variable for total time of the trip in seconds                                      |
| **base_passenger_fare**    | Continuous variable for base fare.                                                             |
| **tolls**                  | Continuous variable for all tolls paid in trip.                                                |
| **bcf**                    | Continuous variable for total amount collected for Black Car Fund.                             |
| **sales_tax**              | Continuous variable for total amount collected for NYS sales tax.                              |
| **congestion_surcharge**   | Continuous variable for total amount collected for NYS congestion surcharge.                   |
| **airport_fee**            | $2.50 for both drop off and pick up at LaGuardia, Newark, and John F. Kennedy airports.        |
| **tips**                   | Continuous variable for total amount of tips received from passenger.                          |
| **driver_pay**             | Continuous variable for total driver pay.                                                      |
| **shared_request_flag**    | Binary variable for shared/pooled ride.                                                        |
| **shared_match_flag**      | Binary variable for if the passenger shared a ride.                                            |
| **access_a_ride_flag**     | Binary variable for if trip was administered on behald of MTA.                                 |
| **wav_request_flag**       | Binary variable for if passenger requested a wheelchair-accessible vehicle.                    |
| **wav_match_flag**         | Binary variable for if the trip occurred in a wheelchair-accessible vehicle.                   |
| **surge_price**            | Binary target variable that will be defined and calculated in the next steps.                  |

Here we can see some quick summary stats of the data:

| Summary    | trip_miles  | trip_time  | base_passenger_fare | tolls      | bcf        | sales_tax  | congestion_surcharge | airport_fee | tips       | driver_pay |
|------------|-------------|------------|---------------------|------------|------------|------------|----------------------|-------------|------------|------------|
| **count**  | 1521319081  | 1521319081 | 1521319081          | 1521319081 | 1521319081 | 1521319081 | 1520806040           | 1106884027  | 1521319081 | 1521319081 |
| **mean**   | 4.9256      | 1156.1248  | 23.0378             | 1.0285     | 0.6308     | 1.9330     | 1.0175               | 0.1983      | 0.9709     | 18.2325    |
| **stddev** | 5.7233      | 830.2167   | 20.8419             | 3.6747     | 0.6331     | 1.7166     | 1.3173               | 0.6790      | 3.0405     | 16.1434    |
| **min**    | 0.0         | 0          | -1969.5900          | 0.0        | 0.0        | -3.0000    | 0.0                  | 0.0         | 0.0        | -6867.2800 |
| **max**    | 5380.7800   | 240764     | 8157.7400           | 1720.0     | 213.0200   | 724.0800   | 13.7500              | 10.0        | 1000.0     | 4894.6200  |

# 3. Exploratory Data Analysis and Visualizations

## 3.1 Surge Conditions - When are surge conditions more likely?
![trips_per_hour](visualizations/trips_per_hour.png)

Taking a look at trips per hour shows us when surge pricing is likely to occur. When the number of trips per hour is increasing, that is a likely scenario when the prices may surge. From the chart it looks like prices will typically be higher around the hours of 5 am - 7 am and again from 10 am - 6 pm.

![trips_per_day_of_week](visualizations/trips_per_day_of_week.png)

This chart maps the number of trips per day, we can see a trend of rising trips as the week goes on. Surge pricing is likely to occur on the days with the highest number of trips. Friday and Saturday when most people are out for the weekend are ideal for increased fares.

## 3.2 Price Vs Distance - Are there unusual prices?
![fare_vs_trip_time](visualizations/fare_vs_trip_time.png)

We can see here that the base passenger fare seems to vary quite a lot regardless of the total time of the trip. This indicates that there may be surge pricing because a short trip can cost the same amount as a much longer trip.

![fare_vs_distance](visualizations/fare_vs_distance.png)

This is similar to the last chart but the fare versus the distance traveled in miles. It seems like there is a similar correlation here where the price can vary regardless of distance.

## 3.3 Demand Vs Price - Is there a relationship here?
![trips_vs_avg_fare_per_mile](visualizations/trips_vs_avg_fare_per_mile.png)

This is the number of trips per hour in a zone versus the average price per hour. What this shows us is that there seems to be a high concentration on the left side where there are the lowest trips per hour but the average price varies quite a bit. This suggests that there may be surge pricing involved because the price is still within the 4-6 dollar range even at 500 trips per hour.

## 3.4 Unique Employer Breakdown - Is there an even distribution or does one company have a majority?
![trips_per_employer](visualizations/trips_per_employer.png)

This is just to see if there is an even distribution of drivers for each employer. It seems like Uber has a significant market share with Lyft coming in second. Juno and Via are almost nonexistent. This kind of control would allow for surge pricing as there are less options from competitors.

# 4. Preprocessing Plan

## 4.1 Missing Data and Feature Engineering

For the numerical variables, there isn't much missing data. The largest dropoff is for the airport fee, which is reasonable since not all trips begin or end at an airport. Most of these numerical columns will be encoded as binary columns for model training. Some sanity checks and cleaning that need to be done involve looking at values that don't make sense and removing them. For example, having a negative base fare or a trip that is over 5,000 miles seem more like errors or outliers. These kinds of extreme values will be removed as they would create noisy points in the training data. The remaining columns will be either dropped or used to generate binary indicators or new columns that may signal surge pricing.

Final set of features:

| Feature                    | Type    | Notes                               |
| -------------------------- | ------- | ------------------------------------|
| `license_index`            | `int`   | encoded platform signal             | 
| `PULocationID`             | `int`   | leave as is, geographical signal    |
| `DOLocationID`             | `int`   | leave as is, geographical signal    |
| `trip_miles`               | `float` | leave as is, trip characteristic    |
| `trip_time`                | `float` | leave as is, trip characteristic    |
| `base_passenger_fare`      | `float` | leave as is, pricing characteristic |
| `tips`                     | `float` | leave as is, pricing characteristic |
| `driver_pay`               | `float` | leave as is, pricing characteristic |
| `has_toll`                 | `int`   | binary fee indicator                |
| `has_airport_fee`          | `int`   | binary fee indicator                |
| `has_congestion_surcharge` | `int`   | binary fee indicator                |
| `hour_of_day`              | `int`   | temporal characteristic             |
| `day_of_week`              | `int`   | temporal characteristic             |
| `month`                    | `int`   | temporal characteristic             |
| `is_weekend`               | `int`   | temporal characteristic             |
| `wait_time_secs`           | `float` | derived demand proxy                |
| `fare_per_mile`            | `float` | derived pricing ratio               |
| `fare_per_min`             | `float` | derived pricing ratio               |
| `driver_pay_ratio`         | `float` | derived pricing ratio               |
| `demand_zscore`            | `float` | derived demand signal               |
| `high_demand`              | `int`   | derived demand signal               |
| `PU_Borough_ID`            | `int`   | encoded geographical signal         |
| `DO_Borough_ID`            | `int`   | encoded geographical signal         |
| `is_surge`                 | `int`   | derived **target label**            |

## 4.2 Duplicate Values

To check for duplicates, since we do not have unique trip identifiers, we can utilize the pickup datetime column. Since there are a maximum of 81 similar pickup datetimes, we can assume there are no duplicate entries in the data. 81 people being picked up at the same time on a random day doesn't seem out of the ordinary for these companies that are completing over 10,000 trips a day. That is less than 1% of the trips being completed on a daily basis. Here are the top 5 counts of matching pickup date times:

| pickup_datetime     | count |
| ------------------- | ----- |
| 2024-11-21 23:06:49 | 81    |
| 2020-03-16 16:55:46 | 65    |
| 2025-02-19 08:00:00 | 65    |
| 2026-01-28 07:30:00 | 64    |
| 2025-02-12 07:30:00 | 63    |

# 4.3 Next Steps
We will be using feature engineering to create an is_surge column to detect the effect of surge pricing. For the rest of the columns we will apply encoding based on their values using techniques like OneHotEncoder and StringIndexer.

For preproccessing we will use Spark operations such as:
- dropna()
- fillna()
- withColumn()
- filter()
- OneHotEncoder()
- groupBy()
- agg()
- and more if needed depending on use case.

# 5. Preprocessing and Feature Engineering

## 5.1 Preprocessing
To clean up erroneous values or outliers, we will check the 99th percentile values for `trip_mile`, `trip_time`, `base_passenfer_fare`, `driver_pay`.

```python
df.approxQuantile(
    ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay"],
    [0.001, 0.01, 0.99, 0.999, 0.9999],
    0.001
)

[0.0, 0.48, 26.09, 5380.78, 5380.78],
[0.0, 193.0, 4094.0, 240764.0, 240764.0],
[-1969.59, 3.75, 104.12, 8157.74, 8157.74],
[-6867.28, 0.0, 77.56, 4894.62, 4894.62]

```
Based on the result, we can see that the 99.9th and 99.99th percentile values are the same as the max values for these features. This means they are most likely erroneous if not outliers and we can safely set a threshold slightly above the 99th percentile for safe measure. This will set an upper bound on the data. After setting these caps and also setting minimum values so that there are no negative values or extremely short trips, the `tips` column seems to still have a somewhat unexpected max value of $500. This was also cleaned up repeating the process for checking the 99th percentile value.

## 5.2 Feature Engineering
**is_surge**
- The first value we want to calculate is the `median_fare_per_mile`, which is the baseline median for each platform + pickup zone + hour + day of the week. This allows us to create a baseline threshold on which to calculate surge quantifiers. Once we calculate the median, we can then create our target label, `is_surge`, by setting a threshold such as 25% above the median and check to see if the ratio of `fare_per_mile`/`median_fare_per_mile` is above or below this threshold.

**high_demand**
- Approximate demand by counting trips per zone within a rolling time window and comparing it to the historical trend. If it's above 1.5 standard deviations, it can be considered as high demand since surge pricing may occur when demand outpaces supply.

**driver_pay_ratio**
- Useful as a way to cross check against the passenger fare. If both are increased then it can indicate surge pricing.

**single riders**
- Our focus will be on single riders only since shared rides may sometimes be discounted or given at reduced rates.

**hvfhs_license_num**
- Map this feature to integer from string since it's only 4 values.

**tolls, airport_fee, congestion_surcharge**
- Map these to binary, we only want to know if these occur, the amounts are standard if they are utilized.

**borough**
- Adding borough identifiers as indices from the location id, helps to create a more generalized geographical signal since location id's might be too fine grained.

After filtering the data and completing feature engineering to derive the necessary columns, we are left with 1,382,815,732 rows which means we lost about 9.1% of our overall data. This is not really an issue since we still have a generous amount of data left.

# 6 First Model - Decision Tree Classifier
## 6.1 Model Setup, Training, and Testing
The first model we will use is a **Decision Tree Classifier (DTC)** since it is easy to decompose and understand if we need to dig into the individual nodes. Hyperparameter tuning is also not as complex as some other models and this model is supported by PySpark.

The first DTC build included all 23 features, with a 70/30 split for training and testing. For the hyperparameters we used a maxDepth of 10 to allow for more complexity and maxBins was set to 265 to match the number of ID's we have in pickup and dropoff locations.

Looking at the counts of our target label `is_surge`, we notice a slight class imbalance:
| is_surge Value | Count       | Percentage |
| -------------- | ----------- | ---------- |
| None           |       2,149 |       0.0% |
| 1              | 402,714,159 |     29.12% |
| 0              | 980,101,573 |     70.88% |

We can safely remove the None rows because the `median_fare_per_mile` was most likely null and missing data. It's a negligible amount of data to lose so it won't affect our outcome. Because there is a slight class imbalance here we can use class weights to allow the model to learn when surge pricing is occurring rather than simply predicting no surge because ~71% of the time that would be the correct answer. The formula for the class weights is as follows:

- weight = total / (num_classes * class_count)

This penalizes the model about 1.72x more for misclassifying a surge trip versus only about a 0.71x penalty for misclassifying a non surge trip. This weight was added as a hyperparameter for weightCol.

After training and testing we have the following results:

```
Total rows: 1,382,815,732

is_surge=1: 402,714,159 (29.12%)
is_surge=0: 980,101,573 (70.88%)

Surge weight: 1.7169
Non-surge weight: 0.7054

AUC-ROC: 0.9310                  This is a strong model, it can distinguish between surge and non surge categories well.
accuracy: 0.9181                 High accuracy at predicting surge.
f1: 0.9190                       Nice balance between precision and recall.
weightedPrecision: 0.9225        The weighted precision value, adjusted for the class weight.
weightedRecall: 0.9181           The weighted recall value, adjusted for the class weight.
```

| Feature Index | Feature                  | Importance |
| ------------: | ------------------------ | ---------- |
|            10 | fare_per_mile            |   0.785562 |
|            16 | hour_of_day              |   0.061470 |
|             2 | PU_borough_index         |   0.042712 |
|            15 | has_congestion_surcharge |   0.037616 |
|             0 | PULocationID             |   0.021402 |
|            19 | is_weekend               |   0.014882 |
|             3 | DO_borough_index         |   0.011870 |
|            11 | fare_per_min             |   0.008050 |
|            14 | has_airport_fee          |   0.007184 |
|             6 | trip_time                |   0.003721 |
|             4 | license_index            |   0.002384 |
|            17 | day_of_week              |   0.001921 |
|             7 | base_passenger_fare      |   0.000362 |
|            13 | has_toll                 |   0.000304 |
|             9 | driver_pay               |   0.000292 |
|             1 | DOLocationID             |   0.000160 |
|            20 | wait_time_secs           |   0.000097 |
|             5 | trip_miles               |   0.000011 |
|             8 | tips                     |   0.000000 |
|            12 | driver_pay_ratio         |   0.000000 |
|            18 | month                    |   0.000000 |
|            21 | demand_zscore            |   0.000000 |
|            22 | high_demand              |   0.000000 |

The feature importance highlights what might be a fundamental issue. 78% of the predictive value is based on `fare_per_mile`. While this is not a straightforward data leak for our target, it may have a big impact since our target is derived from it. As a result of this we will rerun the same model but remove both `fare_per_mile` and `fare_per_min` since that may also dominate if we only removed `fare_per_mile`. Removing these two features will force the model to learn if surge exists from contextual features rather than price signals alone.

The features with zero importance are also worth noting. We derived `demand_zscore` and `high_demand` specifically as surge signals but these had no impact on the model. This can mean that either `fare_per_mile` was so dominant that the model never even needed to utilize these features. After running the model again, we will see if this changes. `month` being zero simply implies that there is no differentiating factor for surge on a month to month basis after controlling for other features.

This run of the DTC doesn't seem to be underfitting or overfitting, but we cannot be entirely confident since we only ran it against the test set. Since we have strong scores for AUC-ROC, accuracy, and F1, it is not underfitting because these scores would all be much lower. Since we have a maxDepth of 10, where the default value is 5, we can be somewhat certain that it is learning more complex and meaningful patterns. The test scores are all consistent so the model is not really overfitting either. It is also not overfitting because we have millions of examples per leaf, making it difficult to memorize noise. However, the best way to check is to generate the training scores and check the gap between training and validation, which we will do in a subsequent run.

## 6.2 Second Run with Decision Tree Classifier
For our second run, hyperparameters will not be tuned initially. `fare_per_mile` and `fare_per_min` have been removed from the list of features and we have also created a validation set for this run. So now we have a 70/15/15 split for training, validation, and testing. This run will have the values for the validation set. Based on the results, we will test some hyperparameter tuning before running against the test set.

```
Validation AUC-ROC: 0.8962
accuracy: 0.8869
f1: 0.8887
weightedPrecision: 0.8964
weightedRecall: 0.8869
```

| Feature Index | Feature                  | Importance |
| ------------- | ------------------------ | ---------- |
|             5 | trip_miles               |   0.628848 |
|             7 | base_passenger_fare      |   0.193749 |
|            10 | driver_pay_ratio         |   0.089184 |
|             2 | PU_borough_index         |   0.032661 |
|            14 | hour_of_day              |   0.028799 |
|            13 | has_congestion_surcharge |   0.014129 |
|             3 | DO_borough_index         |   0.008753 |
|            12 | has_airport_fee          |   0.001152 |
|            17 | is_weekend               |   0.001001 |
|             6 | trip_time                |   0.000801 |
|             0 | PULocationID             |   0.000661 |
|            15 | day_of_week              |   0.000154 |
|             1 | DOLocationID             |   0.000108 |
|            11 | has_toll                 |   0.000000 |
|             9 | driver_pay               |   0.000000 |
|             8 | tips                     |   0.000000 |
|            16 | month                    |   0.000000 |
|             4 | license_index            |   0.000000 |
|            18 | wait_time_secs           |   0.000000 |
|            19 | demand_zscore            |   0.000000 |
|            20 | high_demand              |   0.000000 |

The model still has strong results, we only lost about 3.5% AUC by removing the strongest prior feature. However, there is a trend of pricing signals dominating the feature importance yet again. `trip_miles`, `base_passenger_fare`, and `driver_pay_ratio` are all pricing adjacent features so the model is just learning that longer trips or higher costs means surge is probable. While it is not incorrect, it seems to heavily rely on these pricing signals. This time there are still several features with zero importance, of which our two derived features are still apart of. The model is either heavily relying on pricing signals or it is not able to capture more complex relationships between features at a maxDepth of 10.

Before we attempt to tune the hyperparamenters, we will run it once more. This time we will capture the training and validation scores to check for overfitting and we will remove all of the pricing adjacent features which include `trip_miles`, `base_passenger_fare`, `driver_pay_ratio`, `driver_pay`, as well as `tips` since it is often a percentage of total fare. This will show us whether or not the contextual signals actually are able to predict surge.

## 6.3 Third Run with Decision Tree Classifier
Now we are seeing that the model is learning that longer trips mean surge, which still seems oversimplified. With the Training and Validation AUC scores being the same, it tells us that the model may be underfitting and that with a maxDepth of 10, there isn't enough complexity in the model to achieve more feature interaction. `demand_zscore` is now showing up with some feature importance so the derived signal is meaningful when it's not crowded out by pricing signals. `high_demand` might be too aggressive of a derivation being set at 1.5 standard deviations or above. At this stage we can either allow a higher maxDepth for more complexity and feature interactions, introduce `driver_pay_ratio` back into the mix since that is capturing platform behavior more than pricing but still adjacent, or use a Random Forest Classifier (RFC) with a set number of trees and increased maxDepth try to find a better mix of features for predicing surge. The Random Forest approach may be the better choice since we've lost a meaningful amount of predictive value in this run but it will also take longer to run so we will first try changing maxDepth to 15 to see if we can increase model performance on DTC.

```
Training AUC-ROC: 0.6898
Validation AUC-ROC: 0.6898
Gap: 0.0000
accuracy: 0.7089
f1: 0.7164
weightedPrecision: 0.7620
weightedRecall: 0.7089
```

| Feature Index | Feature                  | Importance |
| ------------- | ------------------------ | ---------- |
|             5 | trip_time                |   0.733820 |
|             3 | DO_borough_index         |   0.112718 |
|             2 | PU_borough_index         |   0.056082 |
|             1 | DOLocationID             |   0.026920 |
|             6 | has_toll                 |   0.017058 |
|            14 | demand_zscore            |   0.013085 |
|             7 | has_airport_fee          |   0.012257 |
|             0 | PULocationID             |   0.012117 |
|            13 | wait_time_secs           |   0.006090 |
|             8 | has_congestion_surcharge |   0.005298 |
|            11 | month                    |   0.002422 |
|             9 | hour_of_day              |   0.001371 |
|             4 | license_index            |   0.000749 |
|            10 | day_of_week              |   0.000015 |
|            12 | is_weekend               |   0.000000 |
|            15 | high_demand              |   0.000000 |

## 6.4 Fourth Run with Decision Tree Classifier

Attempting to run the DTC with maxDepth 15 was not successful at the current time. This section will be updated once errors no longer persist. But in theory, have more depth for the trees would potentially allow more features to have more importance so we may see the values from the last run shift and the models predictive capabilities increase slightly. However, the caveat here is that the more we increase depth, the more the model tends to overfit, so we shouldn't continue to increase the depth even if performance improves.

## 6.5 Conclusion and Next Steps

After removing all of the pricing signals which were dominating the DTC, the contextual features were still mostly dominated by `trip_time` but we saw some of our derived demand signals coming into play. At this stage we will move on to using Ray Train and LightGBM for more complexity without reintroducing the pricing signals back into the mix. While our model did perform well with the pricing signals, we want to test if contextual features can also provide strong predictive power. If we are successful then we can potentially drop `trip_time` and have a model capable of predicting surge pricing based on features that wouldn't require a user to even check the cost but simply know the locations, hour of day, and day of week to know if they will be charged more for their trip.

The total runtime for the entire pipeline was about 3-4 hours. This was possible because of task parallelization and having several executors running at once to process the data, build, and evaluate the models. Trying to do this on a laptop or computer would have crashed with 1.38 billion rows of data. An additional step that will be implemented is writing the cleaned dataset to a parquet file prior to training the models, so that if the model training fails, we won't need to process the data again and can start by reading the cleaned data. This will also be helpful when we use Ray Train to run a LightGBM model a bit later on.

A screenshot of the driver and total memory after running model training (left 1 core and some RAM for overhead):
![spark_screenshot](visualizations/spark_train.png)

Moving forward, we will run a LightGBM (LGBM) model as our next model using Ray Train, which is optimized for machine learning algorithms. DTC creates a single tree with sequential splits, which is a weakness because it only has single model. A single bad split would result in issues all the way down the tree. It is also prone to overfitting at higher depths. With LGBM, similar to XGBoost (XGB) but uses leaf wise growth instead of level wise growth, we have a boosted ensemble method which builds trees sequentially, specifically focusing on picking the most impactful leaf to split. It also has sequential boosting (each model corrects the previous models errors), L1 and L2 regularization, native missing value handling, feature interaction detection which is a key aspect that DTC lacks, and gradient optimization. LGBM is also designed for larger datasets so it will train faster than XGB but with similar results. Setting the `num_leaves` parameter will be important so we do not overfit because leaf wise growth can overfit easily on smaller datasets. However, since this dataset is fairly large, it won't be as much of an issue but still good to keep in mind.

We will also be using dimensionality reduction and testing our second model with both a full set of features as well as the suggested features with dimensionality reduction. It will be interesting to see what features are left after reduction and compare them to the set of features we manually made adjustments for when training the DTC.