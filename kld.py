import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load the CSV file
file_path = 'KLDscores.csv'
kld_scores_df = pd.read_csv(file_path)

# Function to compute the desired measures
def compute_kld_measures(kld_values):
    kld_values = np.array(kld_values)
    avg_kld = np.mean(kld_values)
    var_kld = np.var(kld_values)
    std_kld = np.std(kld_values)
    kld_range = np.max(kld_values) - np.min(kld_values)
    
    # Linear regression to find the slope
    x = np.arange(len(kld_values))
    slope, intercept, r_value, p_value, std_err = linregress(x, kld_values)
    
    return avg_kld, var_kld, std_kld, kld_range, slope

# Apply the function to each row in the dataframe
measures = kld_scores_df['kld_values'].apply(lambda x: compute_kld_measures(eval(x)))

# Create a new dataframe to store the measures
measures_df = pd.DataFrame(measures.tolist(), columns=['avg_kld', 'var_kld', 'std_kld', 'kld_range', 'slope'])
measures_df['filename'] = kld_scores_df['filename']

# Display the new dataframe with the computed measures
print(measures_df.head())
