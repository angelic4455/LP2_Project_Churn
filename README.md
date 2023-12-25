# LP2_Project_Churn
# Predictive Analysis For Customer Retention At Vodafone

## Objective
The objective of this project is to utilize machine-learning models within the CRISP-DM framework to predict customer churn at Vodafone Corporation. The goal is to identify key indicators of churn and propose effective retention strategies based on the analysis. The findings will be compiled into a comprehensive presentation for the business development unit, emphasizing actionable insights to enhance customer retention efforts.

## Hypothesis
In this section, we state our null and alternate hypotheses, along with the analytical questions we seek to answer:

- **Null Hypothesis (H₀):** There is no significant association between a customer's tenure and the likelihood of churn.
- **Alternative Hypothesis (H₁):** A customer's tenure is significantly associated with the likelihood of churn, suggesting that longer tenure reduces the probability of churn.

Analytical Questions:
1. What is our Churning rate?
2. What type of contract churns more customers?
3. What type of internet service churns more customers?
4. Which gender churns more?
5. What is the longest period of time we have had a customer?
6. What is the relationship between tenure and Churn?

# Set Up

## Installation
Here is the section where we installed all the packages/libraries needed to tackle the challenge. The following packages were installed using the provided commands:

```bash
pip install numpy
pip install pandas
pip install patool
pip install forex_python
pip install pandas_profiling

Importations
We imported necessary libraries for this project, including pyodbc, pandas, numpy, matplotlib, seaborn, and scikit-learn. Machine learning libraries and metrics, as well as feature processing tools, were also included.

Establishing a Connection To SQL Server
To connect to the SQL Server, we loaded environment variables from a .env file and used them to create a connection string. The connection was established using the pyodbc library.

Data Loading From Various Sources
Data was loaded from various sources for analysis. The dataset includes information such as customerID, gender, tenure, contract type, monthly charges, and churn status to mention but the few.

# Data Loading From Various Sources (Continued)

## Loading Data from SQL Server

Data was loaded from the SQL Server, specifically the `LP2_Telco_churn_first_3000` table, for analysis. A snippet of the loaded data is provided below:

```python
# Load data from server 
query = 'SELECT * FROM LP2_Telco_churn_first_3000;'
first_data = pd.read_sql(query, connection)
first_data.head()

The loaded data has the following columns:

customerID: Unique identifier for each customer.
gender: Gender of the customer.
SeniorCitizen: Boolean indicating if the customer is a senior citizen.
Partner: Boolean indicating if the customer has a partner.
Dependents: Boolean indicating if the customer has dependents.
tenure: Number of months the customer has stayed with the company.
PhoneService: Boolean indicating if the customer has a phone service.
MultipleLines: Type of phone service (if applicable).
InternetService: Type of internet service.
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: Boolean indicating if the customer has these additional services.
Contract: Type of contract the customer has.
PaperlessBilling: Boolean indicating if the customer has paperless billing.
PaymentMethod: Payment method used by the customer.
MonthlyCharges: Monthly charges paid by the customer.
TotalCharges: Total charges paid by the customer.
Churn: Boolean indicating if the customer has churned.
For more details, you can use the first_data.info() method to check the data types and null counts.

# Check the data info
first_data.info()

Note: The actual code may contain additional details.

# Data Loading From Various Sources (Continued)

## Loading Data from Excel File

The analysis extends to a second dataset, loaded from an Excel file (`Telco-churn-second-2000.xlsx`). Here is a glimpse of the data:

```python
# Replace with the correct relative path to your file
xlsx_file_path = "C:/Users/ALFRED/OneDrive/GitHub/LP2_Project_Churn/Telco-churn-second-2000.xlsx"

# Load the Excel file into a pandas DataFrame
second_data = pd.read_excel(xlsx_file_path)

# Display the first few rows of the DataFrame
second_data.head()

The loaded data has the following columns:

customerID: Unique identifier for each customer.
gender: Gender of the customer.
SeniorCitizen: Integer indicating if the customer is a senior citizen.
Partner, Dependents: Boolean indicating if the customer has a partner or dependents.
tenure: Number of months the customer has stayed with the company.
PhoneService: Boolean indicating if the customer has a phone service.
MultipleLines: Type of phone service (if applicable).
InternetService: Type of internet service.
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: Boolean indicating if the customer has these additional services.
Contract: Type of contract the customer has.
PaperlessBilling: Boolean indicating if the customer has paperless billing.
PaymentMethod: Payment method used by the customer.
MonthlyCharges: Monthly charges paid by the customer.
TotalCharges: Total charges paid by the customer.
For more details, you can use the second_data.info() method to check the data types and null counts.


# Check the data info
second_data.info()

The data includes 2000 entries and 20 columns.

Note: The actual code may contain additional details.

# Data Summary

## Dataset 1
- The dataset contains 2000 entries and 20 columns.
- No missing values found.
- Data types:
  - Float64: 1 column
  - Int64: 2 columns
  - Object: 17 columns

## Dataset 2
- The dataset contains 2000 entries and 20 columns.
- No missing values found.
- Data types:
  - Float64: 1 column
  - Int64: 2 columns
  - Object: 17 columns


## Data Preprocessing

### Checking for NaN values
The initial analysis of the dataset involved checking for missing values. The code snippet below shows the absence of NaN values in the dataset.

```python
# Checking for NaN values
second_data.isna().sum()

The output indicated that there were no missing values in the dataset.

Concatenating DataFrames
After confirming the cleanliness of the data, the next step involved combining two DataFrames, presumably named first_data and second_data. The concatenation was performed along the rows (vertically), and the index was reset for the resulting DataFrame.

# Concatenate along rows (vertically)
concatenated_data = pd.concat([first_data, second_data], axis=0)

# Resetting the index after concatenation
concatenated_data.reset_index(drop=True, inplace=True)

# Display the concatenated DataFrame
concatenated_data

Standardizing Columns
The subsequent task was to standardize categorical columns. As an example, the column 'Partner' was standardized to have values 'Yes' and 'No' instead of boolean values. The code snippets below demonstrate how this was done.

# Standardizing the Partner column
# Replace True with 'Yes' and False with 'No'
concatenated_data['Partner'] = concatenated_data['Partner'].replace({True: 'Yes', False: 'No'})

# Display the updated DataFrame
print(concatenated_data['Partner'].value_counts())

The 'Partner' column was transformed to have 'No' and 'Yes' as its unique values, and the counts of each value were printed for verification.

These preprocessing steps contribute to preparing the dataset for further analysis and modeling.

Dependents Column:

Original values: [False, True, 'No', 'Yes']
After standardization: ['No', 'Yes']
Count: No - 3521, Yes - 1479
PhoneService Column:

Original values: [True, False]
After standardization: ['Yes', 'No']
Count: Yes - 4538, No - 462
MultipleLines Column:

Original values: [None, False, True, 'Yes', 'No', 'No phone service']
After standardization: ['No', 'Yes', 'No phone service']
Count: No - 2403, Yes - 2135, No phone service - 193
InternetService Column:

Original values: ['DSL', 'Fiber optic', 'No']
After standardization: ['Fiber optic', 'DSL', 'No']
Count: Fiber optic - 2191, DSL - 1712, No - 1097
OnlineSecurity Column:

Original values: [False, True, None, 'No', 'No internet service', 'Yes']
After standardization: ['No', 'Yes', 'No internet service']
Count: No - 2469, Yes - 1434, No internet service - 446
DeviceProtection Column:

Original values: [False, True, None, 'No', 'No internet service', 'Yes']
After standardization: ['No', 'Yes', 'No internet service']
Count: No - 2172, Yes - 1731, No internet service - 446
TechSupport Column:

Original values: [False, True, None, 'No', 'No internet service', 'Yes']
After standardization: ['No', 'Yes', 'No internet service']
Count: No - 2477, Yes - 1426, No internet service - 446
StreamingTV Column:

Original values: [False, True, None, 'Yes', 'No internet service', 'No']
After standardization: ['No', 'Yes', 'No internet service']
Count: No - 1982, Yes - 1921, No internet service - 446
SeniorCitizen Column:

Original values: [0, 1]
StreamingMovies Column:

Original values: [False, True, None, 'No', 'No internet service', 'Yes']
After standardization: ['No', 'Yes', 'No internet service']
Count: No - 1954, Yes - 1949, No internet service - 446

Note: The standardization involves replacing True with 'Yes' and False with 'No' in relevant columns. The resulting counts indicate the distribution of values after standardization.

Cell 1:
Display unique values of the 'PaperlessBilling' column.
Cell 2:
Standardize the 'PaperlessBilling' column by replacing True with 'Yes' and False with 'No'.
Display the updated count of unique values.
Cell 3:
Display unique values of the 'Churn' column.
Cell 4:
Standardize the 'Churn' column by replacing True with 'Yes' and False with 'No'.
Display the updated count of unique values.
Cell 5:
Generate descriptive statistics for numerical columns ('SeniorCitizen', 'tenure', 'MonthlyCharges') in the DataFrame.
Cell 6:
Check for duplicate rows in the DataFrame.
Cell 7:
Display the count of missing values for each column in the DataFrame.
Cell 8:
Display the first row of the DataFrame for reference.
This code aims to prepare and clean the data for further analysis.

Cell 9:
Display the first five rows of the DataFrame, providing a glimpse of the dataset.
Observations:
Customer Details:

customerID, gender, SeniorCitizen, Partner, Dependents columns provide information about customers.
tenure represents the duration of the customer's subscription.
PhoneService, MultipleLines, InternetService are related to the services availed by customers.
Service Usage:

Columns like OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies indicate whether the customer uses these services.
Billing and Payment:

Contract, PaperlessBilling, PaymentMethod specify billing and payment details.
MonthlyCharges and TotalCharges are related to financial aspects.
Churn Information:

Churn indicates whether a customer has churned (Yes) or not (No).
Cell 10:
Check for duplicated rows. There are no duplicate rows in the dataset.
Cell 11:
Display the count of missing values for each column in the DataFrame.
Observations:
Missing values are present in columns like MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, TotalCharges, and Churn.
Cell 12:
Display descriptive statistics for numerical columns (SeniorCitizen, tenure, MonthlyCharges) in the DataFrame.
Cell 13:
Check for the presence of duplicate rows in the DataFrame.
Observations:
The DataFrame has no duplicate rows.
Cell 14:
Display the count of missing values for each column in the DataFrame.
Observations:
Missing values are present in columns like MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, TotalCharges, and Churn.
Cell 15:
Display the first row of the DataFrame for reference.
Conclusion:
This code snippet provides a preliminary exploration of the dataset, addressing issues like unique value standardization, duplicate checks, and missing value analysis. Further steps might involve imputing or handling missing values, exploring relationships between variables, and preparing the data for machine learning models or further analysis.


Exploratory Data Analysis
Explore the data to understand patterns, relationships, and potential features that may impact customer churn. Visualizations and statistical analyses can provide insights into the factors influencing customer decisions.

Machine Learning Model Development
1. Data Splitting
To ensure the effectiveness of our machine learning model, we begin by dividing our dataset into two subsets: a training set and a testing set. The training set is used to train the model, while the testing set evaluates its performance on unseen data. This step helps prevent overfitting and provides a realistic assessment of the model's generalization ability.

2. Feature Engineering
Feature engineering involves the creation or transformation of features to improve the model's predictive capabilities. In our context, this might include deriving new insights from existing variables or encoding categorical variables for compatibility with machine learning algorithms. Effective feature engineering enhances the model's ability to capture patterns and make accurate predictions.

3. Model Selection
Choosing the right machine learning algorithm is critical. In our case, since we're dealing with a classification problem (predicting churn or not), we need to select a suitable classification algorithm. Popular choices include logistic regression, decision trees, random forests, and support vector machines. The choice depends on the nature of the data and the problem at hand.

4. Model Training
With the algorithm selected, we proceed to train our model using the training dataset. The model learns from the patterns and relationships within the data, adjusting its parameters to make accurate predictions. Training involves an iterative process of feeding data into the model, evaluating its predictions, and refining its internal settings.

5. Model Evaluation
Once the model is trained, we evaluate its performance on the testing dataset. This step involves using various metrics, such as accuracy, precision, recall, and F1 score, to assess how well the model generalizes to new, unseen data. A thorough evaluation helps us understand the strengths and weaknesses of the model and guides potential improvements.

6. Hyperparameter Tuning
Fine-tuning involves adjusting the hyperparameters of the model to optimize its performance. Hyperparameters are settings that are not learned from the data but influence the learning process. Grid search or random search can be employed to explore different combinations of hyperparameter values. This step is crucial for achieving the best possible model performance.

7. Deployment
Once satisfied with the model's performance, we move on to the deployment phase. Deployment involves integrating the model into a real-world environment where it can make predictions on new data. This could be an operational system, a web application, or any platform where real-time predictions are needed.

By following these steps, we create a robust machine learning model for predicting customer churn, providing valuable insights for business decision-making and proactive customer retention strategies.



Conclusion
Customer churn prediction is crucial for retaining valuable customers. By leveraging machine learning, telecommunications companies can proactively address customer concerns and reduce churn rates. This repository provides a framework for building and deploying a customer churn prediction model.









