import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
import statsmodels.api as sm
from src.config_loader import load_config
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def feature_engineering():
    try:
        # Load the processed data
        val = pd.read_pickle("data/processed/val.pkl")
        eval = pd.read_pickle("data/processed/eval.pkl")
        logger.info("Validation and evaluation data loaded successfully.")
        
        # Extract data for one item that passes all tests
        def cv2(dataWithoutZeros):
            return (np.std(dataWithoutZeros) / np.mean(dataWithoutZeros)) ** 2

        def adi(days, salesWithoutZeros):
            return len(days) / len(salesWithoutZeros)

        def test_all():
            for item in val["id"].unique():
                # Get item dataframe
                item_val = val[val['id'] == item]
                item_val = item_val.reset_index().drop(['index'], axis=1)

                # Check if item is just white noise, random walk or gaussian noise
                data_diff = pd.Series(item_val["Sales"]).diff()
                random_walk = adfuller(data_diff.dropna())
                if random_walk[1] > 0.05:
                    continue
                stat, p = shapiro(data_diff.dropna())
                if p > 0.05:
                    continue
                result = sm.stats.diagnostic.acorr_ljungbox(item_val["Sales"], lags=[10], return_df=False).values[0][1]
                if result > 0.05:
                    continue

                # Check if item is smooth trend
                if cv2(item_val[item_val["Sales"] != 0]["Sales"]) > 0.49:
                    continue
                if adi(item_val["Day"], item_val[item_val["Sales"] != 0]["Sales"]) > 1.32:
                    continue

                # If all tests pass, return item
                return item

        item_name = test_all()

        logger.info(f"Valid item that passed all tests: {item_name}")

        item_val = val[val['id'] == item_name]
        item_val = item_val.reset_index().drop(['index'], axis=1)

        item_eval = eval[eval['id'] == item_name.replace("validation", "evaluation")]
        item_eval = item_eval.reset_index().drop(['index'], axis=1)

        # Make categorical columns into dummies
        item_eval = pd.get_dummies(item_eval, columns=["event_name_1", "event_type_1", "event_name_2", "event_type_2"])
        item_val = pd.get_dummies(item_val, columns=["event_name_1", "event_type_1", "event_name_2", "event_type_2"])

        # Ensure both dataframes have the same columns
        item_val, item_eval = item_val.align(item_eval, join='outer', axis=1, fill_value=0)

        # Convert any non-numeric columns that should be numeric
        numeric_columns = ["sell_price"]
        for column in numeric_columns:
            item_val[column] = pd.to_numeric(item_val[column], errors='coerce')
            item_eval[column] = pd.to_numeric(item_eval[column], errors='coerce')

        # Drop columns with null sell_price
        item_val = item_val.dropna(subset=["sell_price"])

        # Rename columns to match Prophet's requirements
        item_val = item_val.rename(columns={"Day": "ds", "Sales": "y"})
        logger.info("item_val: \n%s", item_val.head())
        item_eval = item_eval.rename(columns={"Day": "ds", "Sales": "y"})
        logger.info("item_eval: \n%s", item_eval.head())

        logger.info("Feature engineering completed successfully.")
        return item_val, item_eval, item_name

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

if __name__ == "__main__":
    item_val, item_eval, item_name = feature_engineering()
