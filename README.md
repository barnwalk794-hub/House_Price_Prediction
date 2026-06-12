# House_Price_Prediction
# 🏠 House Price Prediction

> Predicting house prices using Machine Learning ⚡

## 📌 About

This project predicts house prices based on features like
location, size, and other parameters using Linear Regression.
Helps buyers and sellers estimate fair property prices.

## ✨ Features

- 📂 Load and explore house price dataset
- 🔄 Handle categorical data (Location encoding)
- 🤖 Linear Regression model training
- 📈 Model evaluation with MAE, MSE, RMSE
- 📊 Data visualization with matplotlib

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - Pandas - Data manipulation
  - NumPy - Numerical computing
  - Matplotlib - Data visualization
  - Scikit-learn - Machine learning

## 📁 Dataset Features

| Feature | Description |
|---|---|
| Location | Area/city of the house |
| Price | Target variable (house price) |
| Other features | Size, rooms, etc. |

## 🚀 How to Run

1. Clone the repo:git clone https://github.com/barnwalk794-hub/house-price-prediction

2. Install dependencies:
pip install pandas numpy matplotlib scikit-learn

3. Add your dataset:
Place house_price_data.csv in the project folder

4. Run the script:
python house_price.py

## 📊 Model Performance

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |

## 📈 How it Works

1. Load dataset from CSV
2. Convert Location (categorical) to numbers using One-Hot Encoding
3. Split data 80% train / 20% test
4. Train Linear Regression model
5. Predict house prices
6. Evaluate using MAE, MSE, RMSE

## 👩‍💻 Developer

**Khushi Kumari**
- B.Tech CSE-AI | GITA, Bhubaneswar
- GitHub: github.com/barnwalk794-hub

## ⭐ Show your support

Give a ⭐ if you like this project!

---
*Built with ❤️ using Python and Machine Learning*
