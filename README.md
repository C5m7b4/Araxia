![logo](images/logo.png
)

# 🕸️ Araxia

**Araxia** is a lightweight, customizable time series forecasting model tailored for structured datasets like grocery store sales. Inspired by the structure of spider webs and powered by neural networks, Araxia makes forecasting intuitive, modular, and fun.

> “Pattern is the web. Prediction is the spider.” — *Araxia Philosophy*

---

## 🧠 Features

- 🔁 Lagged features with flexible windowing
- 📅 Time-aware feature engineering (cyclic time, dates, day-of-week)
- 🕸️ Neural network forecaster (built with NumPy from scratch)
- 🧪 Easy backtesting on training data
- 📈 Built-in plot support for forecasts
- 🧩 Modular: plug in your own preprocessing or model layers
- ✅ Lightweight with no heavy dependencies

---

## 🚀 Installation

```bash
pip install araxia
```

Or if you are developing locally:
```bash
git clone https://github.com/your-username/araxia.git
cd araxia
pip install -e .
```

if You want to run the locally installed dev environment
```bash
pip install -e .
```

## 📦 Quickstart

```python
from araxia import create_lagged_dataset, create_dataset_with_cyclic_features
from araxia import AraxiaForecaster
```

### Load your time series data (must include a datetime column and target)
```python
df = your_dataframe[['date', 'sales']]
```

### Create features
```python
df_feat = create_lagged_dataset(df, target='sales', lags=[1, 7, 14])
df_feat = create_dataset_with_cyclic_features(df_feat, date_col='date')
```

### Train-test split
```python
train = df_feat.iloc[:-30]
test = df_feat.iloc[-30:]
```

### Fit model
```python
model = AraxiaForecaster(hidden_sizes=[16, 8], epochs=500, learning_rate=0.01)
model.fit(train.drop(columns='sales'), train['sales'])
```

### Predict
```python
forecast = model.predict(test.drop(columns='sales'))
```

### Plot
```python
model.plot(train['sales'], forecast, title="Araxia Forecast")
```
### 🧰 Core Modules

| Module                                | Purpose                                         |
| ------------------------------------- | ----------------------------------------------- |
| `create_lagged_dataset`               | Generate lag features (e.g., 1-day, 7-day lags) |
| `create_dataset_with_cyclic_features` | Add sine/cosine encodings for day/month/year    |
| `create_dataset_with_date_features`   | Add year, month, day columns                    |
| `create_dataset_with_onehot_dow`      | One-hot encode day-of-week                      |
| `AraxiaForecaster`                    | Custom NumPy-based feedforward network          |


### ⚙️ Customize


Araxia is modular! Replace or enhance the following:

- Use your own preprocessing logic
- Swap out the NumPy model for PyTorch/TensorFlow
- Add early stopping, dropout, or regularization
- Integrate into a pipeline with sklearn

### 🧪 Tests

```python
pytest
```

### 📄 License

MIT License

### 🤝 Contributing

Contributions welcome! Fork the repo, create a branch, and submit a PR.