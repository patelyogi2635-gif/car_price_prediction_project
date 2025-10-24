# 🚗 Car Price Prediction System



A comprehensive machine learning project that predicts car prices based on various specifications using Random Forest Regression. Features an interactive Streamlit web application for real-time predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a machine learning solution to predict used car prices based on 25+ features including engine specifications, performance metrics, physical attributes, and brand information. The system achieves **92% accuracy (R² Score)** using Random Forest Regression with hyperparameter tuning.

### Key Highlights:
- ✅ **Accurate Predictions**: R² Score of 0.92
- ✅ **Interactive Web App**: User-friendly Streamlit interface
- ✅ **Comprehensive Analysis**: EDA with visualizations
- ✅ **Multiple Models**: Comparison of 6 ML algorithms
- ✅ **Production Ready**: Deployable web application

## ✨ Features

### Machine Learning Pipeline:
- Data preprocessing and cleaning
- Feature engineering (brand extraction, age calculation)
- Handling categorical variables (Label & One-Hot Encoding)
- Feature scaling using StandardScaler
- Multiple model training and comparison
- Hyperparameter tuning with GridSearchCV
- Cross-validation for model validation
- Model persistence using pickle

### Web Application:
- 🏠 **Home Page**: Project overview and statistics
- 🔮 **Price Prediction**: Interactive form with organized tabs
- 📊 **Dataset Information**: Feature descriptions and distributions
- 📈 **Model Performance**: Metrics and visualizations
- 💾 **Export Functionality**: Download prediction reports as CSV
- 📱 **Responsive Design**: Works on all devices

## 📊 Dataset

The dataset contains **205 car records** with **26 features**:

### Features Include:
- **Categorical**: CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem
- **Numerical**: symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg
- **Target**: price

### Data Statistics:
- **Price Range**: $5,118 - $45,400
- **Brands**: 20+ manufacturers
- **Body Types**: Sedan, Hatchback, Wagon, Convertible, Hardtop
- **Fuel Types**: Gas, Diesel



### Making Predictions

1. Navigate to the **Predict Price** page
2. Fill in the car specifications across three tabs:
   - Basic Info (brand, fuel type, doors, body type)
   - Engine & Performance (horsepower, engine size, RPM)
   - Dimensions (wheelbase, length, weight, MPG)
3. Click **Predict Price**
4. View the predicted price and breakdown
5. Download the prediction report

## 📈 Model Performance

### Best Model: Random Forest Regressor

| Metric | Score |
|--------|-------|
| **R² Score** | 0.92 |
| **RMSE** | $2,450 |
| **MAE** | $1,850 |
| **Cross-Validation Mean** | 0.92 |

### Model Comparison:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.78 | $4,200 | $3,100 |
| Ridge Regression | 0.80 | $4,000 | $2,950 |
| Lasso Regression | 0.79 | $4,100 | $3,000 |
| Decision Tree | 0.85 | $3,500 | $2,600 |
| **Random Forest** | **0.92** | **$2,450** | **$1,850** |
| Gradient Boosting | 0.90 | $2,800 | $2,100 |

### Top 5 Important Features:
1. **Engine Size** (25%)
2. **Curb Weight** (20%)
3. **Horsepower** (18%)
4. **Car Width** (12%)
5. **Car Length** (8%)

## 📁 Project Structure

```
car-price-prediction/
│
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── car_data.csv                # Dataset (not included)
│
├── models/                     # Saved models
│   ├── car_price_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
│
├── images/                     # Screenshots
│   ├── home_page.png
│   ├── prediction_page.png
│   └── performance_page.png
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore file
```



## 🛠️ Technologies Used

### Machine Learning & Data Science:
- **Python 3.8+**
- **scikit-learn**: Model training and evaluation
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization

### Web Development:
- **Streamlit**: Interactive web application
- **Plotly**: Interactive charts and graphs

### Model Deployment:
- **pickle**: Model serialization
- **joblib**: Alternative serialization

## 🔮 Future Enhancements

- [ ] Add support for multiple datasets
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add batch prediction capability
- [ ] Create REST API using FastAPI
- [ ] Deploy on cloud platforms (Heroku, AWS, GCP)
- [ ] Add user authentication
- [ ] Implement feedback and rating system
- [ ] Add historical price trend analysis
- [ ] Mobile app development
- [ ] Real-time data scraping from car websites



## 👨‍💻 Author

**Your Name**
- GitHub: (https://github.com/patelyogi2635-gif)
- LinkedIn:(https://www.linkedin.com/in/patel-yogi-0a2526346?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- Email: patelyogi2635@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Kaggle Car Price Dataset]
- Inspiration from various ML projects
- Streamlit community for amazing tutorials
- scikit-learn documentation

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- 📧 Email: patelyogi2635@gmail.com
- 💼 LinkedIn: (https://www.linkedin.com/in/patel-yogi-0a2526346?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)


---

⭐ **If you find this project useful, please consider giving it a star!**

---



## 🐛 Bug Reports

Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

## 💡 Feature Requests

Have an idea? Open an issue with the `enhancement` label!

---

**Made with ❤️ and Python**
