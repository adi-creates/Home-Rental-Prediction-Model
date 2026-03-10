# Home Rental Price Predictor

This project trains a **Linear Regression** model using `home-rental.csv` and serves a web UI with Flask to predict monthly rent.

## Features
- Trains a linear regression model using:
  - `postal_code`
  - `size`
  - `bedrooms`
- Validates model performance using a train/validation split
- Displays validation metrics (`R2`, `MAE`, `RMSE`) in the UI
- Provides a web form for rent prediction
- UI inspired by the Microsoft Learning Home Rentals simulator

## Project Structure
```text
Home-Rental/
|-- app.py
|-- home-rental.csv
|-- README.md
|-- templates/
|   `-- index.html
`-- static/
    `-- style.css
```

## Prerequisites
- Python 3.9+

## Install Dependencies
Run this command in the project folder:

```bash
pip install flask pandas numpy
```

## Run the Application
```bash
python app.py
```

Then open your browser at:

```text
http://127.0.0.1:5000
```

## How to Use
1. Select a postal code.
2. Enter property size in square feet.
3. Enter the number of bedrooms.
4. Click **Calculate Rent**.
5. View the predicted rent and model validation metrics.

## Notes
- Model training and validation happen when the Flask app starts.
- Dataset path is expected at `home-rental.csv` in the same folder as `app.py`.
