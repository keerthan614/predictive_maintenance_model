# Predictive Maintenance & Supply Chain Optimization

An end-to-end machine learning project that shifts hardware maintenance from a *reactive* ("fix it when it breaks") model to a *proactive* ("fix it before it breaks") one.

This system analyzes real-time sensor data to predict component failures and then uses those predictions to automatically optimize a global spare parts supply chain, minimizing costs and customer downtime.

## 🚀 Project Overview

This project is a complete, end-to-end solution that solves a critical business problem in two parts:

1.  **Part A: Predictive Maintenance (PdM) Model**

      * **Goal:** Predict when a hardware component is likely to fail.
      * **How:** A hybrid model, led by a high-performance **CNN-LSTM**, is trained on historical sensor telemetry, error logs, and maintenance records. It learns to identify subtle patterns in the data that are leading indicators of a future failure.

2.  **Part B: Supply Chain Optimization Model**

      * **Goal:** Determine the most efficient spare parts inventory allocation.
      * **How:** The failure predictions from Part A are fed into a logistics optimization algorithm. This algorithm calculates the lowest-cost inventory distribution across multiple warehouses to meet the predicted demand, dramatically reducing the need for expensive, last-minute emergency shipments.

## ✨ Key Features

  * **Robust Data Pipeline:** A full pipeline (`pdm_data_processor.py`) that loads, cleans, and engineers 20+ predictive features from 876,000+ real-world telemetry records.
  * **Hybrid Modeling:** Trains, evaluates, and compares 5 different ML models, including:
      * Random Forest
      * Gradient Boosting
      * Logistic Regression
      * LSTM
      * **CNN-LSTM (Best Performer)**
  * **Deep Learning for Time-Series:** Implements a CNN-LSTM architecture in Keras/TensorFlow to effectively capture temporal patterns in sensor data, leading to state-of-the-art predictive accuracy.
  * **Logistics Optimization:** Uses an optimization algorithm to solve the "inventory problem," balancing holding costs against shipping costs to find the optimal stock levels.
  * **Business Impact Analysis:** The pipeline concludes by quantifying its business value, calculating the total net savings and ROI from avoided emergency shipments and proactive maintenance.
  * **End-to-End Execution:** The entire workflow—from raw data to model training, inference, and optimization—is orchestrated by a single script: `pdm_main.py`.

## 💻 Tech Stack

  * **Python**
  * **Pandas & NumPy:** Data manipulation and feature engineering.
  * **Scikit-learn:** For tabular models (Random Forest, GB, LR) and preprocessing.
  * **TensorFlow / Keras:** For building and training the CNN-LSTM and LSTM models.
  * **Matplotlib & Seaborn:** For data visualization and plotting results.
  * **Joblib:** For serializing and saving trained models and data processors.

## 📊 Results & Demo

The models proved highly effective after fixing a data leakage issue. The CNN-LSTM model was the clear winner, successfully identifying complex failure patterns in the sensor data with near-perfect recall.

#### Optimized Inventory Allocation

The final output of the supply chain module, recommending optimal stock levels across warehouses to minimize cost.

#### Feature Importance

The Random Forest model identified key predictors of failure, providing valuable domain insights.

#### Model Training History

The deep learning models converged effectively, demonstrating a clear ability to learn from the sequence data.

## 🛠️ How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2.  **(Recommended) Create and activate a virtual environment:**

    ```bash
    python -m venv pdm-env
    source pdm-env/bin/activate  # On Windows: pdm-env\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main pipeline:**
    This will load all data, process features, train all models, save the best ones, and run the final supply chain optimization demo.

    ```bash
    python pdm_main.py
    ```

## 📂 Project Structure

```
.
├── pdm_main.py           # Main script to run the entire pipeline
├── pdm_data_processor.py # Handles all data loading, cleaning, and feature engineering
├── pdm_models.py         # Defines, trains, and evaluates all ML models
├── pdm_inference.py      # Manages real-time predictions with loaded models
├── pdm_supply_chain.py   # Runs the inventory optimization algorithm
|
├── requirements.txt      # All required Python libraries
|
├── PdM_telemetry.csv     # (All raw data files)
├── PdM_machines.csv
├── PdM_errors.csv
├── PdM_failures.csv
├── PdM_maint.csv
|
├── *.pkl                 # (Saved models and processors)
├── *.keras
|
└── *.png                 # (Generated result images)
```