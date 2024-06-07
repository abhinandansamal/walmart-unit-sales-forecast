import logging
from src.data_preparation import prepare_data
from src.feature_engineering import feature_engineering
from src.model_training import train_and_predict
from src.evaluation import calculate_metrics, quantify_business_impact
from src.submission import create_submission_file
from src.config_loader import load_config
from src.logging_config import setup_logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        # Data Preparation
        logger.info("Starting data preparation...")
        prepare_data()
        logger.info("Data preparation completed successfully.")

        # Feature Engineering
        logger.info("Starting feature engineering...")
        item_val, item_eval, item_name = feature_engineering()
        logger.info("Feature engineering completed successfully.")

        # Model Training and Prediction
        logger.info("Starting model training and prediction...")
        model, item_eval, full_preds = train_and_predict(item_val, item_eval)
        logger.info("Model training and prediction completed successfully.")

        # Evaluate the model
        logger.info("Starting model evaluation...")
        calculate_metrics(item_val, item_eval, full_preds)
        logger.info("Model evaluation completed successfully.")

        # quantifying business impact
        quantify_business_impact(item_val, item_eval)
        logger.info("Business impact quantified successfully.")
        
        # Generate Submission File
        logger.info("Creating submission file...")
        create_submission_file()
        logger.info("Submission file created successfully.")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main()