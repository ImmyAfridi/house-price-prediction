
from pathlib import Path

import joblib
import pandas as pd


def main():
    # 1. Load the saved model
    model_path = Path(__file__).parent / "house_price_model.pkl"
    model = joblib.load(model_path)

    # 2. Enter details for a new house
    house = {
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 1,
        "prefarea": "yes",
        "furnishingstatus": "semi-furnished"
    }

    # 3. Convert the details into a DataFrame
    house_df = pd.DataFrame([house])

    # 4. Predict the price
    predicted_price = model.predict(house_df)[0]

    # 5. Display the result
    print(f"Predicted house price: {predicted_price:,.2f}")


if __name__ == "__main__":
    main()
