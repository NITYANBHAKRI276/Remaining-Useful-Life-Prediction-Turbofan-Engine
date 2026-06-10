# Remaining-Useful-Life-Prediction-Turbofan-Engine
Developed a multi-model ML pipeline (LSTM, XGBoost, Random Forest) on the NASA CMAPSS FD001 dataset to predict the remaining useful life of turbofan jet engines, simulating a Rolls-Royce industrial use case.
Engineered rolling-window features (mean, std over 10 cycles) from 14 sensor streams; clipped RUL targets at 125 cycles to
reduce noise in early-life predictions.
Built a 2-layer stacked LSTM (128→64 units, dropout 0.3) with sequence length 30; achieved RMSE 18.4, MAE 13.2, and R² ≈
0.87 on the test set — outperforming XGBoost (RMSE 19.7) and Random Forest (RMSE 22.1).
Implemented Monte Carlo Dropout inference (50 stochastic passes) to quantify predictive uncertainty; visualised confidence
intervals per engine cycle.
Designed an interactive predictive-maintenance dashboard (HTML/CSS/Chart.js) displaying per-engine RUL, critical/warning/
healthy status, and model-comparison charts.
