# 🍔 Food Delivery Time Prediction (Machine Learning)

🚀 A Machine Learning project to predict food delivery time based on distance, traffic, weather, and other real-world factors.

⭐ Built using Python, Pandas, Scikit-learn, and Data Science techniques

---

## 🎯 Why this Project?

In real-world food delivery platforms like Swiggy and Zomato, predicting delivery time accurately improves:

- Customer satisfaction  
- Delivery efficiency  
- Route optimization  

This project simulates this real-world use case using Machine Learning.

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

The dataset includes factors affecting food delivery time:

- Distance between restaurant and customer  
- Weather conditions  
- Traffic conditions  
- Vehicle type  
- Order details  
- Delivery time (Target variable)  

---

## 🔄 Workflow

Data → Preprocessing → EDA → Model Training → Evaluation → Prediction  

Steps:
1. Data Collection  
2. Data Cleaning  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Model Training  
6. Model Evaluation  
7. Prediction  

---

## 📊 Exploratory Data Analysis (EDA)

Key insights:

- Delivery time increases with distance  
- Traffic condition has strong impact  
- Weather affects delivery delays  
- Vehicle type also influences delivery speed  

---

## 🧠 Model Used

- Linear Regression (Baseline Model)  
- Train-Test Split (80/20)  

### 🚀 Future Improvements:

- Random Forest Regressor  
- XGBoost Model  
- Hyperparameter tuning  
- Deployment as web app  

---

## 📈 Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

> Note: These values may vary depending on dataset split and training conditions.

---

## 📊 Sample Results

| Metric   | Value |
|----------|------|
| MAE      | 6.5  |
| RMSE     | 8.13 |
| R² Score | 0.78 |

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
2. Move into folder
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
