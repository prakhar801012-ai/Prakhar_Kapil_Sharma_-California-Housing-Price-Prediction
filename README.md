# California Housing Price Prediction 🏠📊🤖

A Machine Learning project that predicts California housing prices using Linear Regression and visualizes the relationship between median income and house prices.

## Description

This project uses the California Housing Dataset from Scikit-learn to build a Linear Regression model capable of predicting housing prices based on various housing-related features.

The project also includes data visualization to help understand the relationship between:

- Median Income
- Housing Prices

This is an excellent intermediate-level project for learning:

- Data Analysis
- Data Visualization
- Machine Learning
- Regression Models
- Model Evaluation

---

## Dataset

The project uses the built-in California Housing Dataset provided by Scikit-learn.

### Features

| Feature | Description |
|----------|-------------|
| MedInc | Median Income |
| HouseAge | Median House Age |
| AveRooms | Average Rooms |
| AveBedrms | Average Bedrooms |
| Population | Population |
| AveOccup | Average Occupancy |
| Latitude | Latitude |
| Longitude | Longitude |

### Target Variable

```text
Price
```

Median house value for California districts.

---

## Requirements

Install the required libraries:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

Or from the terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Code

```python
# 1. Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df['Price'] = housing.target

# 3. Quick Data Visualization
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='MedInc',
    y='Price',
    alpha=0.5
)

plt.title('Median Income vs. Housing Price')
plt.show()

# 4. Split Data
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Train Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 6. Assessment Results
y_pred = regressor.predict(X_test)

print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(
    f"Mean Squared Error: "
    f"{mean_squared_error(y_test, y_pred):.4f}"
)
```

---

## How to Run

1. Install Python 3.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Save the code as:

```text
california_housing_prediction.py
```

4. Run the script:

```bash
python california_housing_prediction.py
```

---

## Sample Visualization

The project generates a scatter plot showing the relationship between:

```text
Median Income (MedInc)
        vs
Housing Price
```

This helps identify trends and correlations in the dataset.

---

## Example Output

```text
R2 Score: 0.5758
Mean Squared Error: 0.5559
```

*Results may vary slightly depending on the dataset version and environment.*

---

## Features

✅ California Housing Dataset  
✅ Data Visualization with Seaborn  
✅ Linear Regression Model  
✅ Housing Price Prediction  
✅ Model Performance Evaluation  
✅ Real-World Dataset  
✅ Intermediate Machine Learning Project

---

## Concepts Used

### Data Loading

```python
fetch_california_housing()
```

Loads the California Housing dataset.

---

### Data Visualization

```python
sns.scatterplot()
```

Creates a scatter plot to visualize relationships between variables.

---

### Train-Test Split

```python
train_test_split()
```

Separates data into training and testing sets.

---

### Linear Regression

```python
LinearRegression()
```

Learns the relationship between housing features and prices.

---

### Prediction

```python
regressor.predict()
```

Predicts house prices for unseen data.

---

### Evaluation Metrics

#### R² Score

```python
r2_score()
```

Measures how well the model explains the variance in housing prices.

**Range:**

```text
0 → Poor Fit
1 → Perfect Fit
```

---

#### Mean Squared Error (MSE)

```python
mean_squared_error()
```

Measures the average prediction error.

Lower values indicate better model performance.

---

## Machine Learning Workflow

```text
California Housing Dataset
            ↓
      Data Analysis
            ↓
      Visualization
            ↓
      Train-Test Split
            ↓
     Linear Regression
            ↓
        Prediction
            ↓
   Performance Evaluation
```

---

## Project Structure

```text
california-housing-prediction/
│
├── california_housing_prediction.py
├── README.md
├── requirements.txt
└── housing_visualization.png
```

---

## Skills Demonstrated

- Python Programming
- Data Analysis
- Data Visualization
- Machine Learning
- Linear Regression
- Model Evaluation
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib

---

## Future Improvements

- Add Feature Engineering
- Compare Multiple Algorithms
  - Random Forest
  - XGBoost
  - Decision Trees
- Create Correlation Heatmaps
- Build Interactive Dashboards
- Deploy with Streamlit
- Save and Load Trained Models
- Hyperparameter Tuning

---

## Why This Project Matters

This project demonstrates a complete Machine Learning workflow:

1. Data Loading
2. Data Visualization
3. Data Preparation
4. Model Training
5. Prediction
6. Evaluation

These are the same steps used in many real-world Data Science and AI projects.

---

## License

This project is open source and free to use.
