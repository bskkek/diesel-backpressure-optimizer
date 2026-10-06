"""
Project: AI-Driven Emission & Performance Predictor for Diesel Engines
Institution: Moscow Automobile and Road Construction State Technical University (MADI)
Author: Billel Anisse Bessekek
Year: 2026

Description:
A machine learning framework using Gradient Boosting to predict NOx, Soot emissions,
and Brake Specific Fuel Consumption (BSFC) under variable exhaust backpressure 
and common-rail injection parameters calibrated with Diesel-RK numerical models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

class DieselEmissionAIEngine:
    def __init__(self, n_samples: int = 1500, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.models = {}
        self.metrics = {}

    def generate_synthetic_dataset(self) -> pd.DataFrame:
        """Generates thermodynamic dataset based on calibrated Diesel-RK correlations."""
        np.random.seed(self.random_state)
        
        rpm = np.random.uniform(1000, 2400, self.n_samples)
        load_pct = np.random.uniform(20, 100, self.n_samples)
        delta_p_kpa = np.random.uniform(0, 35, self.n_samples)
        inj_timing_deg = np.random.uniform(8, 20, self.n_samples)
        rail_press_mpa = np.random.uniform(80, 180, self.n_samples)

        # In-cylinder residual gas fraction gamma_r
        gamma_r = 0.035 + (0.00185 * delta_p_kpa) + (0.00001 * rpm / 1000)

        # Empirical emission formulations
        nox_ppm = (180.0 + 8.5 * load_pct + 32.0 * (inj_timing_deg - 10) 
                   - 420.0 * (gamma_r - 0.035) + 1.8 * (rail_press_mpa - 100)
                   + np.random.normal(0, 15, self.n_samples))

        soot_fsn = (0.15 + 0.028 * load_pct + 45.0 * ((gamma_r - 0.035) ** 1.35) 
                    - 0.012 * (inj_timing_deg - 10) - 0.004 * (rail_press_mpa - 100)
                    + np.random.normal(0, 0.05, self.n_samples))
        soot_fsn = np.clip(soot_fsn, 0.05, 5.0)

        bsfc_g_kwh = (205.0 + (1200 / (load_pct + 10)) + 0.65 * delta_p_kpa 
                      - 0.8 * (inj_timing_deg - 14) + np.random.normal(0, 2.5, self.n_samples))

        return pd.DataFrame({
            'rpm': rpm,
            'load_pct': load_pct,
            'delta_p_kpa': delta_p_kpa,
            'inj_timing_btdc': inj_timing_deg,
            'rail_pressure_mpa': rail_press_mpa,
            'nox_ppm': nox_ppm,
            'soot_fsn': soot_fsn,
            'bsfc_g_kwh': bsfc_g_kwh
        })

    def train_and_evaluate(self):
        """Trains Gradient Boosting Regressors for multi-objective engine optimization."""
        df = self.generate_synthetic_dataset()
        features = ['rpm', 'load_pct', 'delta_p_kpa', 'inj_timing_btdc', 'rail_pressure_mpa']
        X = df[features]
        targets = ['nox_ppm', 'soot_fsn', 'bsfc_g_kwh']

        for target in targets:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            
            regressor = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=self.random_state)
            regressor.fit(X_train, y_train)
            
            predictions = regressor.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            self.models[target] = regressor
            self.metrics[target] = {'R2': round(r2, 4), 'MAE': round(mae, 4)}

    def predict_optimal_point(self, rpm: float, load_pct: float, delta_p_kpa: float, inj_timing: float, rail_press: float):
        """Predicts engine behavior for a specific operating condition."""
        input_data = np.array([[rpm, load_pct, delta_p_kpa, inj_timing, rail_press]])
        return {
            'Predicted NOx (ppm)': round(float(self.models['nox_ppm'].predict(input_data)[0]), 2),
            'Predicted Soot (FSN)': round(float(self.models['soot_fsn'].predict(input_data)[0]), 3),
            'Predicted BSFC (g/kWh)': round(float(self.models['bsfc_g_kwh'].predict(input_data)[0]), 2)
        }

if __name__ == "__main__":
    print("=" * 70)
    print("MADI Scientific ML Pipeline: AI Engine for Diesel Emission Prediction")
    print("Author: Bessekek Billel Anisse | Group 1MDVS, MADI")
    print("=" * 70)
    
    engine = DieselEmissionAIEngine()
    engine.train_and_evaluate()
    
    print("\nModel Training Performance Metrics (Test Set):")
    for metric, vals in engine.metrics.items():
        print(f" -> Target: {metric:<12} | R² Score: {vals['R2']} | MAE: {vals['MAE']}")

    sample_prediction = engine.predict_optimal_point(rpm=1800, load_pct=75, delta_p_kpa=20, inj_timing=14, rail_press=140)
    print("\nSimulation Case Study (1800 RPM, 75% Load, 20 kPa Backpressure):")
    for k, v in sample_prediction.items():
        print(f"   {k}: {v}")
