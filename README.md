# 🚀 CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

## 📌 Project Overview

This project focuses on predicting Bitcoin prices using deep learning models across multiple future time horizons (1-day, 3-day, and 7-day).

Bitcoin price movements are highly volatile and exhibit complex temporal patterns. Traditional models struggle to capture these dependencies, so this project leverages **deep learning sequence models** to improve forecasting accuracy.

---

## 🎯 Objectives

* Perform **time-series forecasting** on Bitcoin price data
* Implement **multi-horizon prediction**:

  * 1-Day Forecast
  * 3-Day Forecast
  * 7-Day Forecast
* Compare deep learning models:

  * CNN
  * RNN
  * LSTM
* Evaluate performance using standard regression metrics

---

## 📊 Dataset

The dataset contains historical Bitcoin market data with features:

* Date
* Price (Target Variable)
* Open
* High
* Low
* Volume
* Change %

---

## ⚙️ Data Preprocessing

* Cleaned numeric columns (removed commas, %, K/M/B formats)
* Handled missing values using forward/backward fill
* Sorted data by date (time-series requirement)
* Applied **MinMax Scaling** separately for:

  * Input features
  * Target variable
* Created sequences using sliding window:

  * Input: last 60 days
  * Output: next N days (1, 3, 7)

---

## 🧠 Models Implemented

### 🔹 CNN (1D Convolutional Neural Network)

* Captures short-term patterns
* Fast training
* Weak for long-term dependencies

### 🔹 RNN (Recurrent Neural Network)

* Handles sequential data
* Better than CNN for time-series

### 🔹 LSTM (Long Short-Term Memory)

* Captures long-term dependencies
* Solves vanishing gradient problem
* **Best performing model in this project**

---

## 📈 Results (Best Model: LSTM)

| Horizon | MAE  | RMSE  | MAPE   |
| ------- | ---- | ----- | ------ |
| 1-Day   | 5236 | 12503 | 12.88% |
| 3-Day   | 5852 | 13916 | 14.53% |
| 7-Day   | 7008 | 13727 | 17.69% |

---

## 🔍 Key Insights

* LSTM significantly outperformed CNN and RNN
* Forecast accuracy decreases as prediction horizon increases
* Bitcoin volatility makes long-term prediction difficult
* CNN fails to capture long-term temporal dependencies
* RNN improves but still limited compared to LSTM

---

## 📉 Visualizations

* Training vs Validation Loss Curves
* Actual vs Predicted Price Graphs
* Horizon-wise performance comparison

---

## 🏗️ Project Structure

```bash
bitcoin-price-forecasting-dl/
│
├── data/
│   └── raw/bitcoin.csv
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/Monisha-12/bitcoin-price-forecasting-dl.git
cd bitcoin-price-forecasting-dl
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Project

```bash
python main.py
```

---

## 🔄 Model Selection

Modify in `main.py`:

```python
model_name = "cnn"   # cnn / rnn / lstm
horizon = 1          # 1 / 3 / 7
```

---

## 🏁 Conclusion

This project demonstrates the effectiveness of deep learning models in financial time-series forecasting. Among all models, **LSTM achieved the best performance** due to its ability to capture long-term dependencies.

As prediction horizon increases, accuracy decreases due to increased uncertainty in volatile markets like cryptocurrency.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## 👩‍💻 Author

**Monisha Selvaraj**
Frontend & Mobile Developer → Transitioning to AI/ML

---
