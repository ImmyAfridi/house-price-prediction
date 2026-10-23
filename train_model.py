
from pathlib import Path
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)


def main():
    # 1. Locate and load the dataset
    data_path = Path(__file__).parent / "Housing.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Place Housing.csv in the project folder."
        )

    df = pd.read_csv(data_path)

    # 2. Separate features (X) from the target (y)
    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    df = df.dropna(subset=["price"])

    X = df.drop(columns=["price"])
    y = df["price"]

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # 3. Split data before fitting preprocessing steps
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
    # Define five-fold cross-validation
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 4. Define preprocessing for each feature type
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # 5. Define the models
    models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

        # 1. Compare models using five-fold cross-validation
    cv_results = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        fold_mae = -scores
        mean_mae = fold_mae.mean()
        std_mae = fold_mae.std()

        cv_results[model_name] = mean_mae

        print(f"\n--- {model_name} ---")
        print(f"Fold MAEs: {fold_mae.round(2)}")
        print(f"Mean MAE:  {mean_mae:,.2f}")
        print(f"Std MAE:   {std_mae:,.2f}")

    # 2. Rank models by cross-validation MAE
    print("\n--- Cross-Validation Ranking ---")

    for name, mae in sorted(
        cv_results.items(),
        key=lambda item: item[1]
    ):
        print(f"{name}: Mean MAE = {mae:,.2f}")

    best_model_name = min(
        cv_results,
        key=cv_results.get
    )

    print(
        f"\nSelected model by cross-validation MAE: "
        f"{best_model_name}"
    )

    # 3. Evaluate all models on the untouched test set
    results = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[model_name] = {
            "MAE": mae,
            "R2": r2
        }

        print(f"\n--- Test Results: {model_name} ---")
        print(f"MAE: {mae:,.2f}")
        print(f"R²:  {r2:.3f}")

    # 4. Display test results ranked by MAE
    print("\n--- Test Set Comparison ---")

    for name, metrics in sorted(
        results.items(),
        key=lambda item: item[1]["MAE"]
    ):
        print(
            f"{name}: MAE = {metrics['MAE']:,.2f}, "
            f"R² = {metrics['R2']:.3f}"
        )

    # 5. Retrain the selected model on all training data
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[best_model_name])
    ])

    best_pipeline.fit(X_train, y_train)

    # 6. Save the selected pipeline
    model_path = Path(__file__).parent / "house_price_model.pkl"
    joblib.dump(best_pipeline, model_path)

    print(f"\nSelected model: {best_model_name}")
    print(f"Cross-validation MAE: {cv_results[best_model_name]:,.2f}")
    print(f"Test MAE: {results[best_model_name]['MAE']:,.2f}")
    print(f"Test R²: {results[best_model_name]['R2']:.3f}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
