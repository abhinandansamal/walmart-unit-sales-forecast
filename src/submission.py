import pandas as pd
import logging
from src.config_loader import load_config
from src.logging_config import setup_logging
from src.feature_engineering import feature_engineering
from src.model_training import train_and_predict

setup_logging()
logger = logging.getLogger(__name__)

def load_future_dates():
    try:
        future_dates = pd.read_pickle("data/processed/future.pkl")
        return future_dates
    except Exception as e:
        logger.error(f"Error loading future dates: {e}")
        raise

def create_submission_file():
    try:
        # Load config
        config = load_config()
        
        # Feature engineering to get item_name and item_val
        item_val, item_eval, item_name = feature_engineering()
        
        # Model training to get model and item_eval
        model, item_eval, full_preds = train_and_predict(item_val, item_eval)

        # Predict future sales using future.pkl
        future_dates = load_future_dates()
        future_dates = future_dates[future_dates['id'] == item_name.replace("validation", "evaluation")]

        # Ensure future_dates have the same regressors as the training data
        future_dates = pd.get_dummies(future_dates, columns=["event_name_1", "event_type_1", "event_name_2", "event_type_2"])
        future_dates = future_dates.reindex(columns=item_val.columns, fill_value=0)

        # Make predictions on future dates
        future_preds = model.predict(future_dates)
        future_dates = future_dates.merge(future_preds[['ds', 'yhat']], on='ds')

        # Display future predictions
        logger.info(f"Future predictions: \n{future_dates[['ds', 'yhat']].head()}")

        # Generate submission file
        sample_submission = pd.read_csv(config['data_paths']['sample_submission'])
        submission = sample_submission.copy()
        submission_cols = ["id"] + [f"F{i}" for i in range(1, 29)]

        # Create predictions for the submission file
        future_pred_vals = future_dates.set_index('ds')['yhat'].values[:28]
        submission.loc[submission['id'] == item_name.replace("validation", "evaluation"), submission_cols[1:]] = future_pred_vals

        # Save submission
        submission_file_path = config['submission']['file_path']
        submission.to_csv(submission_file_path, index=False)
        logger.info(f"Submission file created successfully at {submission_file_path}")

    except Exception as e:
        logger.error(f"Error creating submission file: {e}")
        raise

if __name__ == "__main__":
    create_submission_file()
