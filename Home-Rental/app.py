from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "home-rental.csv"
FEATURE_COLUMNS = ["size", "bedrooms", "postal_code"]
TARGET_COLUMN = "rent_amount"


class LinearRegressionModel:
    """Simple linear regression using least squares."""

    def __init__(self) -> None:
        self.coefficients = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_values = x.to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=float)
        x_design = np.c_[np.ones(len(x_values)), x_values]
        self.coefficients = np.linalg.lstsq(x_design, y_values, rcond=None)[0]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction.")
        x_values = x.to_numpy(dtype=float)
        x_design = np.c_[np.ones(len(x_values)), x_values]
        return x_design @ self.coefficients


def train_validation_split(
    x: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_data = x.copy()
    split_data[TARGET_COLUMN] = y
    shuffled = split_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_index = int(len(shuffled) * (1 - test_size))

    train_data = shuffled.iloc[:split_index]
    val_data = shuffled.iloc[split_index:]

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]
    x_val = val_data[FEATURE_COLUMNS]
    y_val = val_data[TARGET_COLUMN]
    return x_train, x_val, y_train, y_val


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    return {
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
    }


def load_and_train_model(data_path: Path) -> dict:
    """Load rental data, train a linear regression model, and compute validation metrics."""
    data = pd.read_csv(data_path)

    x = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    x_train, x_val, y_train, y_val = train_validation_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegressionModel()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    metrics = regression_metrics(y_val.to_numpy(dtype=float), y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "postal_codes": sorted(data["postal_code"].astype(str).unique().tolist()),
        "size_min": int(data["size"].min()),
        "size_max": int(data["size"].max()),
        "size_default": int(data["size"].median()),
        "bedroom_min": int(data["bedrooms"].min()),
        "bedroom_max": int(data["bedrooms"].max()),
        "bedroom_default": int(data["bedrooms"].median()),
    }


MODEL_STATE = load_and_train_model(DATA_PATH)


@app.route("/", methods=["GET", "POST"])
def home() -> str:
    prediction = None
    error = None

    selected_postal_code = MODEL_STATE["postal_codes"][0]
    selected_size = MODEL_STATE["size_default"]
    selected_bedrooms = MODEL_STATE["bedroom_default"]

    if request.method == "POST":
        try:
            selected_postal_code = request.form.get("postal_code", selected_postal_code)
            selected_size = int(request.form.get("size", selected_size))
            selected_bedrooms = int(request.form.get("bedrooms", selected_bedrooms))

            features = pd.DataFrame(
                [
                    {
                        "size": selected_size,
                        "bedrooms": selected_bedrooms,
                        "postal_code": int(selected_postal_code),
                    }
                ]
            )

            predicted_rent = MODEL_STATE["model"].predict(features)[0]
            prediction = round(float(predicted_rent), 2)
        except (ValueError, TypeError):
            error = "Please enter valid input values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        metrics=MODEL_STATE["metrics"],
        postal_codes=MODEL_STATE["postal_codes"],
        selected_postal_code=str(selected_postal_code),
        selected_size=selected_size,
        selected_bedrooms=selected_bedrooms,
        size_min=MODEL_STATE["size_min"],
        size_max=MODEL_STATE["size_max"],
        bedroom_min=MODEL_STATE["bedroom_min"],
        bedroom_max=MODEL_STATE["bedroom_max"],
    )


if __name__ == "__main__":
    app.run(debug=True)
