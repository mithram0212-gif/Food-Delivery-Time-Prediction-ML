# 🍔 Food Delivery Time Prediction (Machine Learning)

🚀 This project predicts food delivery time using machine learning based on distance, traffic, weather, and other real-world factors.

⭐ Built using Python, Pandas, Scikit-learn, and Data Science techniques

---

## 🎯 Project Objective

The objective of this project is to build a machine learning model that can accurately predict food delivery time based on real-world influencing factors such as:

- Distance between restaurant and customer  
- Traffic conditions  
- Weather conditions  
- Vehicle type  

This helps improve delivery efficiency and customer satisfaction in real-world food delivery systems.

---

## 🛠 Technologies Used

- Python 🐍  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Flask  

---

## 📊 Dataset Description

The dataset contains features that influence delivery time:

- Distance between restaurant and customer  
- Weather conditions  
- Traffic conditions  
- Vehicle type  
- Order details  
- Delivery time (Target variable)  

---

## 🔄 Workflow

1. Data Collection  
2. Data Cleaning  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Model Training  
6. Model Evaluation  
7. Prediction  

---

## 📊 Exploratory Data Analysis (EDA)

Key insights from data:

- Delivery time increases as distance increases  
- Traffic condition significantly impacts delivery time  
- Weather conditions cause delays  
- Vehicle type affects delivery speed  

---

## 🧠 Machine Learning Model

- Linear Regression (Baseline Model)  
- Train-Test Split (80/20)  

### 🚀 Future Improvements

- Random Forest Regressor  
- XGBoost Model  
- Hyperparameter tuning  
- Model deployment using Flask  

---

## 📈 Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

> Note: Results may vary depending on dataset split and training conditions.

---

## 📊 Sample Results

| Metric   | Value |
|----------|------|
| MAE      | 6.5   |
| RMSE     | 8.13  |
| R² Score | 0.78  |

---

## 📁 Project Structure


Food-Delivery-Time-Prediction-ML/
│
├── app.py
├── food_delivery.py
├── delivery time prediction.ipynb
├── food_dataset.csv
├── requirements.txt
├── README.md
│
├── images/
│ ├── actual_vs_predicted.png
│ ├── feature_importance_random_forest.png
│ ├── delivery_time_distribution.png
│ ├── avg_delivery_time_by_vehicle.png
│ ├── avg_delivery_time_by_weather.png
│ ├── avg_delivery_time_by_traffic.png
│ ├── Food Delivery Time Prediction UI.png


---

## 🚀 How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/mithram0212-gif/Food-Delivery-Time-Prediction-ML.git
2. Move into project folder
cd Food-Delivery-Time-Prediction-ML
3. Install dependencies
pip install -r requirements.txt
4. Run the project
python food_delivery.py

OR

python app.py
👨‍💻 Author

SANGAMITHRA M

📌 Note

This project is built for academic and learning purposes. Results may vary depending on dataset and model training conditions.

⭐ Show Support

If you like this project, give it a ⭐ on GitHub!
