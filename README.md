# Walmart Sales Forecasting

## Description

This project aims to forecast the daily sales for the next 28 days at Walmart stores using hierarchical sales data. The data covers stores in three US states (California, Texas, and Wisconsin) and includes item-level, department, product categories, and store details. In addition, it contains explanatory variables such as price, promotions, day of the week, and special events. The primary goal is to improve forecast accuracy using machine learning techniques.

## Task Description

Forecasting sales at the item-store level can significantly impact inventory management, reducing overstock and stockouts, and thereby improving business efficiency. In this competition, you are challenged to use both traditional forecasting methods and machine learning to achieve this goal.

## Evaluation

The model's performance is evaluated using the Weighted Root Mean Squared Scaled Error (RMSSE). The primary objective is to minimize this error metric.

## Dataset Description

• **calendar.csv:** Contains information about the dates on which the products are sold.

• **sales_train_validation.csv:** Contains the historical daily unit sales data per product and store [d_1 - d_1913].

• **sales_train_evaluation.csv:** Includes sales [d_1 - d_1941] (labels used for the Public leaderboard).

• **sell_prices.csv:** Contains information about the price of the products sold per store and date.

• **sample_submission.csv:** The correct format for submissions.

## Usage

### Data Preprocessing
The data preprocessing steps are defined in the src/data_preprocessing.py module. This includes loading the data, handling missing values, and performing initial transformations.

### Feature Engineering
Feature engineering steps are implemented in the src/feature_engineering.py module. This involves creating new features that are essential for improving the forecasting model.

### Model Training
The model training script is in the src/model_training.py module. Here, the Prophet model is trained using the preprocessed and engineered features.

### Model Evaluation
Model evaluation is performed in the src/evaluation.py module. This includes calculating the RMSSE and other relevant metrics to assess the model's performance.


## Example Jupyter Notebook

An example Jupyter Notebook is provided in the `notebooks/` directory (`walmart_unit_sales_forecasting.ipynb`). This notebook demonstrates the entire workflow from data loading, preprocessing, feature engineering, model training, and evaluation.


## Project Structure

```bash
walmart-unit-sales-forecast/
├── config
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── calendar.csv
│   │   ├── sales_train_validation.csv
│   │   ├── sample_submission.csv
│   │   ├── sell_prices.csv
│   │   └── sales_train_evaluation.csv
│   ├── processed/
│   │   ├── val.pkl
│   │   ├── eval.pkl
│   │   ├── future.pkl
│
├── logs/
│   └── app.log
│
├── notebooks/
│   └── walmart_unit_sales_forecasting.ipynb
│
├── models/
│   └── prophet_model.joblib
│
├── submissions/
│   └── submission.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── submission.py
│   ├── config_loader.py
│   └── logging_config.py
│
├── main.py
├── environment.yml
├── README.md
├── LICENSE
├── setup.py
└── requirements.txt (optional, if needed for pip packages only)
```

• `data/raw/`: Contains the raw input files (sales data, product prices, etc.).

• `data/processed/`: Contains processed data files (these will be generated during the execution).

• `models/`: Contains saved model files.

• `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and model development.

• `src/`: Contains the source code for data preparation, feature engineering, model training, and evaluation.

• `submissions/`: Contains the submission files for the competition.


## Setup Instructions

### Prerequisites

• Python 3.10.12

• Anaconda installed

• Git installed


### Setup Using Conda

1. Clone the Repository

```bash
git clone https://github.com/abhinandansamal/walmart-unit-sales-forecast.git
cd walmart-sales-forecasting
```

2. Download the data:

    • Download the required raw data files from this [Google Drive](https://drive.google.com/drive/folders/1eXjXk48Zlt52aPlDqACMkdDkEAnlUgGT?usp=sharing) link and place them in the data/raw/ directory. As most of the files are > 50 MB, so they are not uploaded in the repository.

    • Alternatively, you can download the complete dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) here.

3. Create the Conda Environment

```bash
conda env create -f environment.yml
```

3. Activate the Environment

```bash
conda activate walmart_sales_forecasting
```

4. Run the Project

```bash
python main.py
```

### Important Notes

• The processed data files (eval.pkl, future.pkl, val.pkl) will be generated during the execution of main.py. These files are required for the model to make predictions and will stored in the data/processed/ directory. As the file sizes are > 50 MB, so they are not uploaded in the repository. 

• Due to the high RAM requirements for executing this project, it is advisable to run the Jupyter notebook (walmart_unit_sales_forecast.ipynb located in the notebooks/ folder) on a cloud platform or Google Colab with the high RAM option enabled. Alternatively, use a local system with sufficient memory.

• If running the Jupyter notebook in Google Colab, you can upload the raw data files to the Colab environment and adjust the file paths accordingly.

## Notable Enhancements

### Model Training and Evaluation
• **Model:** Implemented Facebook Prophet for time series forecasting.

• **Error Metrics:** Calculated Mean Absolute Error (MAE) and RMSSE to evaluate model performance.

• **Visualizations:** Generated visualizations to show sales trends and forecast accuracy.

### Quantifying Business Impact
• **Savings:** Estimated potential savings of around $250.44 by reducing overstock and stockouts, based on average item price and forecast accuracy.

### Future Work
• **Feature Integration:** Plan to integrate additional features such as marketing campaigns, weather data, and competitor pricing.

• **Algorithm Experimentation:** Explore different machine learning algorithms and ensemble methods to improve accuracy.

• **Hyperparameter Tuning:** Perform hyperparameter tuning to optimize model performance.

• **Extended Forecasting:** Extend the model to predict sales for multiple items and stores.

## Key Findings
• **Model Accuracy:** The model achieved an RMSSE of 0.6054 on the validation set.

• **Business Impact:** By implementing this forecasting model, potential savings of approximately $250.44 can be realized through better inventory management.

## Conclusion

This notebook demonstrates the process of forecasting sales using historical data and the Prophet model. The approach outlined here can be applied to other items and stores to enhance inventory management and business planning.

## Acknowledgments

• This project is based on data provided by UNIVERSITY OF NICOSIA and is available on Kaggle.

• Special thanks to the developers of Prophet and the Python community for their support and contributions to the libraries used in this project.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/abhinandansamal/walmart-unit-sales-forecast/blob/main/LICENSE) file for details.