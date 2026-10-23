# House Price Prediction

A machine learning project that predicts house prices using property features such as area, bedrooms, bathrooms, parking, and furnishing status.

## Project Overview

This project explores and compares four regression algorithms to predict house prices:

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor

Five-fold cross-validation is used to select the model with the lowest average Mean Absolute Error (MAE).

## Dataset

The project uses the Housing Prices dataset from Kaggle by `yasserh`.

Dataset: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

The dataset contains 545 rows and 13 columns, including the target variable `price`.

## Evaluation

The model is selected using five-fold cross-validation on the training data.

| Model | Cross-validation MAE |
|---|---:|
| Linear Regression | 747,208.83 |
| Gradient Boosting | 762,329.74 |
| Random Forest | 765,725.00 |
| Decision Tree | 1,063,698.12 |

Linear Regression was selected because it achieved the lowest average cross-validation MAE.

## Project Structure

- `download_dataset.py` — Downloads the dataset.
- `explore_data.py` — Explores the dataset.
- `train_model.py` — Trains, evaluates, and saves the selected model.
- `predict.py` — Predicts the price of a new house.
- `Housing.csv` — Dataset file (not committed to Git).
- `house_price_model.pkl` — Saved model (generated locally).

## Requirements

- Python
- pandas
- scikit-learn
- kagglehub
- joblib

## Run the Project

Install the dependencies:

```bash
python -m pip install pandas scikit-learn kagglehub joblib
```

Download the dataset using:

```bash
python download_dataset.py
```

Place `Housing.csv` in the project directory, then train the model:

```bash
python train_model.py
```

Predict a house price:

```bash
python predict.py
```

## Limitations

Predictions are estimates based on the patterns in the training data. They are not guaranteed market valuations. Performance may vary for houses with characteristics that differ from the dataset.

## License and Attribution

The dataset belongs to its original publisher. Refer to the Kaggle dataset page for its terms of use.
