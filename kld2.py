import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load the data
kld_scores = pd.read_csv('KLDscores.csv')
metadata = pd.read_csv('SPGC-metadata-2018-07-18.csv')
extra_controls = pd.read_csv('extra_controls.csv')

# Ensure all entries in 'kld_values' are lists
def parse_kld_values(kld_values):
    if isinstance(kld_values, str):
        try:
            return eval(kld_values)
        except:
            return []
    elif isinstance(kld_values, (list, np.ndarray)):
        return list(kld_values)
    else:
        return []

kld_scores['kld_values'] = kld_scores['kld_values'].apply(parse_kld_values)

# Flatten the kld_values for aggregation functions
def flatten_kld_values(kld_values):
    flattened = []
    for sublist in kld_values:
        if isinstance(sublist, list):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened

# Calculate book-level measures of KLD
kld_measures = kld_scores.groupby('filename').agg(
    kld_mean=('kld_values', lambda x: np.mean(flatten_kld_values(x))),
    kld_variance=('kld_values', lambda x: np.var(flatten_kld_values(x)))
).reset_index()

# Calculate the slope of KLD across the narrative
def calculate_slope(kld_values):
    if len(kld_values) > 1:
        X = np.arange(len(kld_values)).reshape(-1, 1)
        y = np.array(kld_values)
        model = LinearRegression().fit(X, y)
        return model.coef_[0]
    else:
        return 0  # Return 0 slope for books with insufficient KLD values

kld_scores['kld_slope'] = kld_scores['kld_values'].apply(lambda x: calculate_slope(flatten_kld_values(x)))
kld_measures = kld_measures.merge(kld_scores[['filename', 'kld_slope']], on='filename')

# Merge with metadata and extra controls
metadata.rename(columns={'id': 'filename'}, inplace=True)
extra_controls.rename(columns={'id': 'filename'}, inplace=True)
data = metadata.merge(kld_measures, on='filename').merge(extra_controls, on='filename')

# Handle missing values by dropping rows with NaNs
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Filter out books with zero or negative downloads
data = data[data['downloads'] > 0]

# Ensure the data is not empty before proceeding
if data.empty:
    raise ValueError("The dataset is empty after processing. Please check the data and preprocessing steps.")

# Prepare the data for regression
data['log_downloads'] = np.log(data['downloads'])

# List of independent variables
independent_vars = ['kld_mean', 'kld_variance', 'kld_slope'] + extra_controls.columns[1:].tolist()

# Prepare data for regression
X = data[independent_vars]
y = data['log_downloads']

# Ensure the features are not empty
if X.empty:
    raise ValueError("The feature matrix X is empty after processing. Please check the data and preprocessing steps.")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add constant for statsmodels
X_sm = sm.add_constant(X_scaled)

# Perform OLS regression
ols_model = sm.OLS(y, X_sm).fit()
print(ols_model.summary())

# Investigate heterogeneity across genres
genres = data['subjects'].unique()

for genre in genres:
    genre_data = data[data['subjects'].str.contains(genre, regex=False)]
    if not genre_data.empty and len(genre_data) > 1:
        X_genre = genre_data[independent_vars]
        y_genre = genre_data['log_downloads']
        
        # Standardize the features
        X_genre_scaled = scaler.transform(X_genre)
        
        # Add constant for statsmodels
        X_genre_sm = sm.add_constant(X_genre_scaled)
        
        # Perform OLS regression
        genre_model = sm.OLS(y_genre, X_genre_sm).fit()
        print(f"Genre: {genre}")
        print(genre_model.summary())

# Perform LASSO regression to infer which variables are most predictive
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# Get coefficients and identify most predictive variables
lasso_coefficients = pd.Series(lasso.coef_, index=independent_vars)
important_vars = lasso_coefficients[lasso_coefficients != 0].index.tolist()

print("Variables selected by LASSO:")
print(important_vars)
