from flask import Flask, request, render_template 
import numpy as np
from sklearn.svm import SVR 
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sqlite3
# Note: Ensure all packages are listed in requirements.txt

# Define the database and model files
DATABASE = 'fcr_data.db' 
MODEL_FILE = 'model.pkl'

# --- Expanded Feed Formulation Data (Nutritional Targets with Proportions) ---
FORMULATION_TARGETS = {
    # ------------------ DAIRY CATTLE ------------------
    'DAIRY COW (Lactating)': {
        'Protein': '16-19%',
        'Energy (TDN)': '75-78%',
        'Fiber (ADF)': '18-21%',
        'Basis': '100 kg Total Feed',
        'Ingredients': [
            ('Corn Silage', '55.0 kg'),
            ('Alfalfa Hay', '25.0 kg'),
            ('Soybean Meal', '17.0 kg'),
            ('Mineral/Vitamin Premix', '3.0 kg')
        ]
    },
    'DAIRY CALF (Starter)': {
        'Protein': '20-22%',
        'Energy (TDN)': '80%',
        'Fiber (ADF)': '10-15%',
        'Basis': '100 kg Total Feed',
        'Ingredients': [
            ('Milk Replacer', '50.0 kg'),
            ('Calf Starter Grain', '45.0 kg'),
            ('Hay (small amounts)', '5.0 kg')
        ]
    },
    
    # ------------------ SWINE (PIGS) ------------------
    'FINISHING PIG (Market)': {
        'Protein': '14-16%',
        'Energy (TDN)': '85%',
        'Fiber (ADF)': '<5%',
        'Basis': '100 kg Total Feed',
        'Ingredients': [
            ('Corn/Barley', '68.0 kg'),
            ('Soybean Meal', '25.0 kg'),
            ('Fat/Oil Source', '4.0 kg'),
            ('Mineral/Vitamin Premix', '3.0 kg')
        ]
    },
    'GROWER PIG': {
        'Protein': '18-20%',
        'Energy (TDN)': '80%',
        'Fiber (ADF)': '5-7%',
        'Basis': '100 kg Total Feed',
        'Ingredients': [
            ('Corn', '60.0 kg'),
            ('Soybean Meal', '35.0 kg'),
            ('Lysine/DCP Supplement', '5.0 kg')
        ]
    },
    
    # ------------------ POULTRY ------------------
    'LAYING HEN (Production)': {
        'Protein': '16-18%',
        'Energy (TDN)': '60%',
        'Fiber (ADF)': '3-5%',
        'Basis': '100 kg Total Feed',
        'Ingredients': [
            ('Corn/Wheat', '58.0 kg'),
            ('Soybean Meal', '20.0 kg'),
            ('Calcium Source (Limestone)', '15.0 kg'),
            ('Mineral/Vitamin Pack', '7.0 kg')
        ]
    }
}
# --- Database Initialization (Runs once when app starts) ---
def init_db(app):
    """Initializes the database and creates the prediction table."""
    # Use app.root_path to ensure the DB file is created in the correct place
    db_path = os.path.join(app.root_path, DATABASE)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight REAL NOT NULL,
            temperature REAL NOT NULL,
            predicted_fcr REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# --- Analysis Function ---
def get_analysis_data(app):
    """Fetches all predictions and calculates summary statistics."""
    db_path = os.path.join(app.root_path, DATABASE)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT weight, temperature, predicted_fcr, timestamp FROM predictions ORDER BY timestamp DESC")
    all_predictions = cursor.fetchall()
    
    conn.close()
    
    # ... (Rest of analysis logic remains the same) ...
    data_list = [{'weight': w, 'temp': t, 'fcr': f, 'time': ts} 
                 for w, t, f, ts in all_predictions]
    
    if data_list:
        fcr_values = np.array([d['fcr'] for d in data_list])
        
        summary = {
            'count': len(data_list),
            'avg_fcr': np.mean(fcr_values),
            'min_fcr': np.min(fcr_values),
            'max_fcr': np.max(fcr_values),
            'std_fcr': np.std(fcr_values)
        }
    else:
        summary = None
        
    return summary, data_list


# --- The Application Factory ---
def create_app():
    app = Flask(__name__)
    
    # --- Model Loading / Training Logic ---
    data = {
        'features': [
            [50, 20], [55, 21], [60, 22], [45, 19], [70, 23],
            [65, 20], [52, 25], [58, 18], [48, 24], [75, 21]
        ],
        'target': [
            2.5, 2.4, 2.6, 2.3, 2.7,
            2.55, 2.8, 2.35, 2.75, 2.6
        ]
    }
    X = np.array(data['features'])
    y = np.array(data['target'])
    
    # Use app.root_path for file locations in production
    model_path = os.path.join(app.root_path, MODEL_FILE)

    if os.path.exists(model_path):
        print(f"Loading trained model and scalers from {MODEL_FILE}")
        with open(model_path, 'rb') as file:
            saved_objects = pickle.load(file) 
            model = saved_objects['model']
            X_scaler = saved_objects['X_scaler']
            y_scaler = saved_objects['y_scaler']
    else:
        # NOTE: In production, this block runs only once during the first deployment
        print("Model file not found. Starting training and hyperparameter tuning...")
        
        X_scaler = StandardScaler() 
        y_scaler = StandardScaler() 
        X_scaled = X_scaler.fit_transform(X)
        y_reshaped = y.reshape(-1, 1)
        y_scaled = y_scaler.fit_transform(y_reshaped)

        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']}
        grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_scaled, y_scaled.ravel())
        
        model = grid_search.best_estimator_
        print(f"Tuning complete. Best parameters found: {grid_search.best_params_}")
        
        saved_objects = {'model': model, 'X_scaler': X_scaler, 'y_scaler': y_scaler}
        with open(model_path, 'wb') as file:
            pickle.dump(saved_objects, file)
        print(f"Best model and scalers saved to {MODEL_FILE}")


    # --- Prediction Function (Nested in factory to access model/scalers) ---
    def predict_fcr(weight, temperature):
        new_data = np.array([[weight, temperature]])
        new_data_scaled = X_scaler.transform(new_data)
        predicted_fcr_scaled = model.predict(new_data_scaled)
        
        predicted_fcr_reshaped = predicted_fcr_scaled.reshape(-1, 1)
        predicted_fcr = y_scaler.inverse_transform(predicted_fcr_reshaped)[0][0]
        return predicted_fcr

    # --- Database Initialization ---
    with app.app_context():
        init_db(app)

    # --- 3. Flask Routes ---
    
    @app.route('/')
    def home():
        return render_template('index.html', result=None)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            weight_str = request.form['weight']
            temp_str = request.form['temp']
            if not weight_str or not temp_str: raise KeyError 

            weight = float(weight_str)
            temp = float(temp_str)
        
        except KeyError:
            error_message = "Please fill in BOTH Animal Weight and Ambient Temperature."
            return render_template('index.html', result=None, error=error_message)

        except ValueError:
            error_message = "Please enter valid NUMERIC values for Weight and Temperature."
            return render_template('index.html', result=None, error=error_message)

        # Prediction
        predicted_value = predict_fcr(weight, temp)
        
        # Database Saving
        db_path = os.path.join(app.root_path, DATABASE)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (weight, temperature, predicted_fcr) VALUES (?, ?, ?)",
            (weight, temp, predicted_value)
        )
        conn.commit()
        conn.close()

        # Recommendation
        recommendation = "FCR is high. Consider adjusting diet composition or checking for heat stress." if predicted_value > 2.6 else "FCR is within an acceptable range for current conditions."
            
        result_data = {'weight': weight, 'temp': temp, 'fcr': f'{predicted_value:.3f}', 'recommendation': recommendation}
        return render_template('index.html', result=result_data)

    @app.route('/formulation')
    def formulation_page():
        animal_types = list(FORMULATION_TARGETS.keys())
        return render_template('formulation.html', animal_types=animal_types, result=None)

    @app.route('/formulate', methods=['POST'])
    def formulate():
        try:
            animal_type = request.form['animal_type']
            targets = FORMULATION_TARGETS.get(animal_type, None)
            if targets:
                result_data = {'type': animal_type, 'targets': targets}
            else:
                return formulation_page(error="Invalid animal type selected.")

            animal_types = list(FORMULATION_TARGETS.keys())
            return render_template('formulation.html', animal_types=animal_types, result=result_data)

        except KeyError:
            error_message = "Please select an animal type."
            animal_types = list(FORMULATION_TARGETS.keys())
            return render_template('formulation.html', animal_types=animal_types, error=error_message)

    @app.route('/analysis')
    def data_analysis():
        summary, all_predictions = get_analysis_data(app)
        
        if summary:
            summary_display = {
                'Count': summary['count'],
                'Average Predicted FCR': f"{summary['avg_fcr']:.3f}",
                'Minimum Predicted FCR': f"{summary['min_fcr']:.3f}",
                'Maximum Predicted FCR': f"{summary['max_fcr']:.3f}",
                'Standard Deviation': f"{summary['std_fcr']:.3f}"
            }
        else:
            summary_display = None
            
        return render_template('analysis.html', summary=summary_display, predictions=all_predictions)

    # --- Crucial: Return the configured app instance ---
    return app

# ---tumakbo ka na ---
# --- Local Runner for Development ONLY ---
if __name__ == '__main__':
    # When running locally, call the factory to get the app
    local_app = create_app()
    local_app.run(debug=True)