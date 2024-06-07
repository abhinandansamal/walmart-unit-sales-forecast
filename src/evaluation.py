import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
from src.config_loader import load_config
from src.logging_config import setup_logging
from src.feature_engineering import feature_engineering
from src.model_training import train_and_predict

setup_logging()
logger = logging.getLogger(__name__)

def calculate_metrics(item_val, item_eval, preds):
    try:
        # Calculate baseline error
        zeros = np.zeros(len(item_eval))
        baseline_mae_zeros = mean_absolute_error(item_eval['y'], zeros)
        logger.info(f"Baseline error, 0s: {baseline_mae_zeros}")

        # Calculate the mean of the data
        mean_val = np.mean(item_val['y'])
        # Calculate MAE
        baseline_mae_mean = mean_absolute_error(item_eval['y'], np.full(len(item_eval), mean_val))
        logger.info(f"Baseline error, mean: {baseline_mae_mean}")

        # Calculate model MAE
        model_mae = mean_absolute_error(item_eval['y'], preds['yhat'])
        logger.info(f"Mean absolute error: {model_mae}")

        # Calculate RMSSE
        def calculate_rmsse(y_true, y_pred, train_series):
            numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
            denominator = np.sqrt(np.mean(np.diff(train_series) ** 2))
            return numerator / denominator

        rmsse_val = calculate_rmsse(item_eval['y'], preds['yhat'], item_val['y'])
        logger.info(f"Validation RMSSE: {rmsse_val:.4f}")

        return baseline_mae_zeros, baseline_mae_mean, model_mae, rmsse_val
    except Exception as e:
        logger.error(f"Error in calculating metrics: {e}")
        raise

def quantify_business_impact(item_val, item_eval):
    try:
        # Calculate average price
        average_price = item_val["sell_price"].mean()
        logger.info(f"Average price: ${average_price:.2f}")

        overstock_reduction = np.sum(item_eval['y'] - item_eval['yhat'][item_eval['y'] > item_eval['yhat']])
        stockout_reduction = np.sum(item_eval['yhat'] - item_eval['y'][item_eval['yhat'] > item_eval['y']])

        potential_savings = (overstock_reduction + stockout_reduction) * average_price
        logger.info(f"Potential Savings from Improved Forecasting: ${potential_savings:.2f}")

        return potential_savings
    except Exception as e:
        logger.error(f"Error in quantifying business impact: {e}")
        raise

if __name__ == "__main__":
    item_val, item_eval, item_name = feature_engineering()
    model, item_eval, full_preds = train_and_predict(item_val, item_eval)

    # Calculate and log metrics
    metrics = calculate_metrics(item_eval, item_val, full_preds)
    logger.info(f"Metrics: {metrics}")

    # Quantify and log business impact
    business_impact = quantify_business_impact(item_val, item_eval)
    logger.info(f"Business Impact: {business_impact}")
