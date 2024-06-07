import os
import numpy as np
import pandas as pd
import logging
from src.config_loader import load_config
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_data():
    config = load_config()
    try:
        calendar = pd.read_csv(config['data_paths']['calendar'])
        sales_train_eval = pd.read_csv(config['data_paths']['sales_train_evaluation'])
        sell_prices = pd.read_csv(config['data_paths']['sell_prices'])
        return calendar, sales_train_eval, sell_prices
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_data():
    calendar, sales_train_eval, sell_prices = load_data()

    # Preprocess sell_prices
    sell_prices["id"] = sell_prices["item_id"] + "_" + sell_prices["store_id"] + "_evaluation"
    sell_prices = sell_prices[["id", "wm_yr_wk", "sell_price"]]

    # Preprocess calendar
    calendar.drop(calendar.columns[2:6], axis=1, inplace=True)

    # Preprocess sales_train_eval
    preformatted_sales_train_eval = sales_train_eval.drop(sales_train_eval.columns[1:6], axis=1)
    formatted_sales_train_eval = pd.melt(preformatted_sales_train_eval, id_vars=['id'], var_name='Day', value_name='Sales')
    del preformatted_sales_train_eval

    # Merge with calendar and sell_prices
    formatted_sales_train_eval = pd.merge(formatted_sales_train_eval, calendar, left_on='Day', right_on='d', how='left')
    formatted_sales_train_eval.drop(["d"], inplace=True, axis=1)
    formatted_sales_train_eval = pd.merge(formatted_sales_train_eval, sell_prices, left_on=['id', 'wm_yr_wk'], right_on=['id', 'wm_yr_wk'], how='left')
    formatted_sales_train_eval.drop(["date", "wm_yr_wk"], inplace=True, axis=1)
    formatted_sales_train_eval["Day"] = formatted_sales_train_eval["Day"].str[2:]
    formatted_sales_train_eval['Day'] = formatted_sales_train_eval['Day'].astype(int)
    formatted_sales_train_eval['Day'] = pd.to_datetime(formatted_sales_train_eval['Day'], origin='2011-01-28', unit='D')

    # Split into val and eval sets
    val = formatted_sales_train_eval[formatted_sales_train_eval["Day"] < "2016-04-25"]
    eval = formatted_sales_train_eval[formatted_sales_train_eval["Day"] >= "2016-04-25"]

    # Rename id column appropriately
    val.loc[:, "id"] = val["id"].str.replace("evaluation", "validation")

    # Save to pickle
    logger.info("Saving validation and evaluation sets to pickle files.")
    val.to_pickle("data/processed/val.pkl")
    eval.to_pickle("data/processed/eval.pkl")

    # Prepare future dataframe
    calendar['date'] = pd.to_datetime(calendar['date'])
    dates = pd.date_range(start="2016-05-23", end="2016-06-19")
    ids = pd.DataFrame(np.repeat(sell_prices["id"].unique(), 28), columns=["id"])
    ids["date"] = np.tile(dates, len(sell_prices["id"].unique()))
    eval_dates = ids
    eval_dates = pd.merge(eval_dates, calendar, left_on='date', right_on='date', how='left')
    eval_dates = pd.merge(eval_dates, sell_prices, left_on=['id', 'wm_yr_wk'], right_on=['id', 'wm_yr_wk'], how='left')
    eval_dates.drop(["d", "wm_yr_wk"], inplace=True, axis=1)
    eval_dates.rename(columns={"date": "ds"}, inplace=True)

    eval_dates.to_pickle("data/processed/future.pkl")
    logger.info("Data preparation completed successfully.")

if __name__ == "__main__":
    prepare_data()
