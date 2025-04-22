import numpy as np
import pandas as pd
import warnings 
from lifelines.utils import concordance_index

def create_dataset(log_ph: callable, baseline_hazard=0.01, ranges=[-1, 1], n_samples=10000, seed=42, return_cindex=False):
    """
    Create survival dataset with given log partial hazard function.

    This function assumes a constant baseline hazard and a uniform censoring distribution.

    Args:
    -----
        log_ph : callable
            Log partial hazard function. The function should take as input the covariates and return the log partial hazard.
        baseline_hazard : float, optional
            Baseline hazard rate. The default is 0.01.
        ranges : list
            List of ranges for the covariates. If a single range is given, it is used for all covariates. The default is [-1, 1].
        n_samples : int
            Number of samples. The default is 10000.
        seed : int
            Random seed.
        return_cindex : bool
            Whether to return the concordance index of the dataset. The default is False.

    Returns:
    --------
        df : pd.DataFrame
            DataFrame with covariates, duration, and event columns.
    """
    np.random.seed(seed)
    covariates = log_ph.__code__.co_varnames

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * len(covariates)).reshape(len(covariates),2)
    else:
        ranges = np.array(ranges)
    
    # Generate covariates and log partial hazard
    x = [np.random.uniform(*x_range, n_samples) for x_range in ranges]
    lph = log_ph(*x)

    # Generate survival times
    hazard = baseline_hazard * np.exp(lph)
    survival_times = np.random.exponential(1 / hazard)
    censoring_times = np.random.uniform(0, survival_times.max(), n_samples)
    observed_times = np.minimum(survival_times, censoring_times)

    # Create dataset
    df = pd.DataFrame({c: x[i] for i, c in enumerate(covariates)})
    df['duration'] = observed_times
    df['event'] = (survival_times <= censoring_times).astype(int)

    # Calculate concordance index using true log partial hazard
    cindex = concordance_index(df['duration'], -lph, df['event'])

    # Print warning if concordance index is low
    if cindex < 0.6: warnings.warn(
        f'Concordance index is low ({cindex:.04f}), which indicates that the covariates have \
            low predictive power. Consider changing the log partial hazard function.')
        
    print(f"Concordance index of true expression: {cindex:.04f}")

    if return_cindex: return df, cindex
    
    return df

if __name__ == "__main__":
    def linear_function(x1, x2,x3,x4):
        return x1 + np.sin(2 * np.pi * x2)

    def quadratic_function(x1, x2,x3,x4):
        return x1 ** 2 + np.sin(2 * np.pi * x2)

    # Example usage
    def log_partial_hazard(x1, x2):
        return 0.5 * x1 + 0.3 * x2

    df = create_dataset(quadratic_function, baseline_hazard=0.01, ranges=[[-1, 1], [-1, 1], [-1, 1], [-1, 1]], n_samples=10000)

    # split 80:20 into test and train
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # save to csv in the data folder
    data_loc = '/Users/mcgoug01/Downloads/CoxKAN/CoxKAN/data/'
    train_df.to_csv(data_loc + 'sim_reviewer_1_3_train.csv', index=False)
    test_df.to_csv(data_loc + 'sim_reviewer_1_3_test.csv', index=False)
    print(df.head())