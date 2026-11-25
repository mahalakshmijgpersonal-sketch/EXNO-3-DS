## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy import stats

print("Libraries imported successfully")

data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

print("Dataset loaded successfully")
data.head()
```
<img width="1092" height="270" alt="image" src="https://github.com/user-attachments/assets/fb3c0a1e-cbd7-4e2f-8b26-64e7e5b9165b" />

```
print("Missing values before cleaning:")
print(data.isnull().sum())

# Fill missing numerical values
for col in data.select_dtypes(include=np.number).columns:
    data[col] = data[col].fillna(data[col].mean())

# Fill missing categorical values
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing values after cleaning:")
print(data.isnull().sum())
```
<img width="592" height="762" alt="image" src="https://github.com/user-attachments/assets/2dd211f8-9aae-47cf-a7e8-b039223283bb" />

```
le = LabelEncoder()

for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col].astype(str))

print("Feature Encoding Completed")
data.head()
```
<img width="1100" height="272" alt="image" src="https://github.com/user-attachments/assets/5d9eece4-ec4c-4b96-8262-fe3f6cd1414a" />
```
# Fill missing fare values
data['fare'] = data['fare'].fillna(data['fare'].mean())
```
```
# Keep positive fare values
fare_data = data[data['fare'] > 0].copy()
```
```
# Log Transformation
fare_data['log_fare'] = np.log(fare_data['fare'])
```
```
# Reciprocal Transformation
fare_data['reciprocal_fare'] = 1 / fare_data['fare']
```
```
# Square Root Transformation
fare_data['sqrt_fare'] = np.sqrt(fare_data['fare'])
```
```
# Square Transformation
fare_data['square_fare'] = np.square(fare_data['fare'])
```
```
# Box-Cox Transformation
fare_data['boxcox_fare'], _ = stats.boxcox(fare_data['fare'])
```
```
# Yeo-Johnson Transformation
pt = PowerTransformer(method='yeo-johnson')
fare_data['yeojohnson_fare'] = pt.fit_transform(fare_data[['fare']])

print("Feature Transformation Completed")
fare_data.head()
```
<img width="1282" height="300" alt="image" src="https://github.com/user-attachments/assets/ef9ae2c5-247c-4e09-b700-a727dfd90954" />
```
fare_data.to_csv('feature_transformed_data.csv', index=False)
print("File saved successfully!")
```
<img width="407" height="37" alt="image" src="https://github.com/user-attachments/assets/261e44c4-340d-43cd-95d5-e6e601b89da8" />
```
print("Final Transformed Dataset:")
display(fare_data.head())

print("\nDataset Shape:", fare_data.shape)
print("\nFinal Columns:")
print(fare_data.columns)
```
<img width="1328" height="528" alt="image" src="https://github.com/user-attachments/assets/da80fba7-802c-465f-b173-65746437020b" />


# RESULT:
      Thus the program to implement the linear regression using gradient descent is written and verified using Python programming.

       
