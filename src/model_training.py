import pandas as pd
from prophet import Prophet
import logging
import joblib
from src.config_loader import load_config
from src.logging_config import setup_logging
from src.feature_engineering import feature_engineering


setup_logging()
logger = logging.getLogger(__name__)

def train_and_predict(item_val: pd.DataFrame, item_eval: pd.DataFrame):
    try:
        # Ensure item_eval is a DataFrame
        if not isinstance(item_eval, pd.DataFrame):
            raise ValueError("item_eval should be a pandas DataFrame")
        
        # Fit model
        model = Prophet()
        # Exclude the 'id' column when adding regressors
        regressor_columns = item_val.columns.difference(['ds', 'y', 'id'])

        # For loop to add all regressors
        for regressor in regressor_columns:
            model.add_regressor(regressor)

        # Fit the model
        model.fit(item_val)

        # Save the model to the models/ folder
        joblib.dump(model, "models/prophet_model.joblib")
        logger.info("Model saved successfully at models/prophet_model.joblib")

        # Predict on eval set
        full_preds = model.predict(item_eval)
        item_eval = item_eval.merge(full_preds[['ds', 'yhat']], on='ds')
        

        return model, item_eval, full_preds
    except Exception as e:
        logger.error(f"Error in training and prediction: {e}")
        raise

if __name__ == "__main__":
    item_val, item_eval, item_name = feature_engineering()
    train_and_predict(item_val, item_eval)
