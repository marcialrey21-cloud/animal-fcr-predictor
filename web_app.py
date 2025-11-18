from flask import Flask, request, render_template 
import numpy as np
from sklearn.svm import SVR 
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from thefuzz import fuzz
from thefuzz import process
import sqlite3
# Note: Ensure all packages are listed in requirements.txt

# Define the database and model files
DATABASE = 'fcr_data.db' 
MODEL_FILE = 'model.pkl'

# --- Expanded Feed Formulation Data (Nutritional Targets with Proportions) ---
# --- Nutritional Content and Cost Data (Example) ---
# --- FORMULATION_TARGETS (Numeric Constraints based on Philippine Recommends) ---
# NOTE: All min/max values are PROPORTIONS (e.g., 0.18 = 18%)
FORMULATION_TARGETS = {
    # ------------------ SWINE (PIGS) - PIC/BAI Standards ------------------
    'GROWER PIG (25-50 kg)': {
        'Min_Protein': 0.16, 'Max_Protein': 0.18,
        'Min_TDN': 0.78, 'Max_TDN': 0.85, 
        'Min_ADF': 0.00, 'Max_ADF': 0.06,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Rice Bran (D1)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Rice Bran (D1)': 0.30}, # Max 30% Rice Bran
    },
    'FINISHER PIG (75-100 kg)': {
        'Min_Protein': 0.13, 'Max_Protein': 0.15,
        'Min_TDN': 0.82, 'Max_TDN': 0.90,
        'Min_ADF': 0.00, 'Max_ADF': 0.04,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Rice Bran (D1)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Rice Bran (D1)': 0.40}, # Max 40% Rice Bran
    },

    # ------------------ POULTRY - BAI/NRC Standards ------------------
    'BROILER CHICK (Starter)': {
        'Min_Protein': 0.22, 'Max_Protein': 0.24, # High protein requirement
        'Min_TDN': 0.85, 'Max_TDN': 0.90,
        'Min_ADF': 0.00, 'Max_ADF': 0.03,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Fish Meal (Local)', 'Limestone (Calcium)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Fish Meal (Local)': 0.08}, # Max 8% Fish Meal
    },
    'LAYING HEN (Production)': {
        'Min_Protein': 0.16, 'Max_Protein': 0.18,
        'Min_TDN': 0.70, 'Max_TDN': 0.75,
        'Min_ADF': 0.03, 'Max_ADF': 0.07,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Rice Bran (D1)', 'Limestone (Calcium)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Limestone (Calcium)': 0.10}, # Max 10% Limestone
    },
    
    # ------------------ RUMINANTS (BEEF/DAIRY) ------------------
    'BEEF CATTLE (Finisher)': {
        'Min_Protein': 0.12, 'Max_Protein': 0.14,
        'Min_TDN': 0.70, 'Max_TDN': 0.75,
        'Min_ADF': 0.25, 'Max_ADF': 0.35, # Requires high fiber
        'Ingredients': ['Corn Silage', 'Soybean Meal (44%)', 'Yellow Corn', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Corn Silage': 0.70}, # Max 70% Silage
    },
    'DAIRY COW (Lactating)': {
        'Min_Protein': 0.17, 'Max_Protein': 0.19,
        'Min_TDN': 0.75, 'Max_TDN': 0.80, 
        'Min_ADF': 0.18, 'Max_ADF': 0.21,
        'Ingredients': ['Corn Silage', 'Alfalfa Hay', 'Soybean Meal (44%)', 'Yellow Corn', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Corn Silage': 0.60, 'Yellow Corn': 0.40}, # Max limits on energy and forage
    },
}

INGREDIENT_DATA = {
    # ENERGY SOURCES
    'Yellow Corn': {'Cost_USD_kg': 0.25, 'Protein': 0.08, 'TDN': 0.88, 'ADF': 0.03},
    'Rice Bran (D1)': {'Cost_USD_kg': 0.15, 'Protein': 0.12, 'TDN': 0.70, 'ADF': 0.07},
    'Cassava Meal': {'Cost_USD_kg': 0.18, 'Protein': 0.02, 'TDN': 0.75, 'ADF': 0.04},
    
    # PROTEIN SOURCES
    'Soybean Meal (44%)': {'Cost_USD_kg': 0.55, 'Protein': 0.44, 'TDN': 0.75, 'ADF': 0.08},
    'Fish Meal (Local)': {'Cost_USD_kg': 0.80, 'Protein': 0.55, 'TDN': 0.78, 'ADF': 0.02},
    'Coconut Meal (Copra)': {'Cost_USD_kg': 0.30, 'Protein': 0.20, 'TDN': 0.65, 'ADF': 0.12},
    
    # FORAGE/SUPPLEMENTS
    'Alfalfa Hay': {'Cost_USD_kg': 0.15, 'Protein': 0.18, 'TDN': 0.55, 'ADF': 0.35},
    'Corn Silage': {'Cost_USD_kg': 0.08, 'Protein': 0.08, 'TDN': 0.60, 'ADF': 0.30},
    'Limestone (Calcium)': {'Cost_USD_kg': 0.05, 'Protein': 0.00, 'TDN': 0.00, 'ADF': 0.00},
    'DCP/Lysine Premix': {'Cost_USD_kg': 1.20, 'Protein': 0.00, 'TDN': 0.00, 'ADF': 0.00},
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

# C:\FlaskAppTest\web_app.py (New Solver Function)

from pulp import LpProblem, LpMinimize, LpVariable, value

def least_cost_formulate(animal_type, target_batch_kg=100):
    """
    Solves for the least-cost feed mix meeting nutritional targets 
    for a given animal type using Linear Programming.
    """
    targets = FORMULATION_TARGETS.get(animal_type)
    if targets is None:
        # If the animal type doesn't exist in the dictionary, return None immediately.
        return None 
    
    # Get the list of ingredients to be used for this animal (defined in the targets)
    ING_KEYS = targets['Ingredients'] # <--- This variable MUST be defined after the check!
    
    # --- 2. Setup the Linear Problem ---
    prob = LpProblem("Least Cost Feed Mix", LpMinimize)
    
    # Decision Variables: Quantity (kg) of each ingredient to use
    x = LpVariable.dicts("Quantity", ING_KEYS, 0)
    
    # 3. Objective Function: Minimize Total Cost
    prob += sum([INGREDIENT_DATA[i]['Cost_USD_kg'] * x[i] for i in ING_KEYS]), "Total Cost"
    
    # 4. Constraints
    prob += sum([x[i] for i in ING_KEYS]) == target_batch_kg, "Total Weight"
    # a) protein constraints (Min and Max)
    prob += sum([INGREDIENT_DATA[i]['Protein'] * x[i] for i in ING_KEYS]) >= target_batch_kg * targets['Min_Protein'], "Min Protein"
    prob += sum([INGREDIENT_DATA[i]['Protein'] * x[i] for i in ING_KEYS]) <= target_batch_kg * targets['Max_Protein'], "Max Protein"

    # b) TDN (Energy) constraints (Min and Max)
    prob += sum([INGREDIENT_DATA[i]['TDN'] * x[i] for i in ING_KEYS]) >= target_batch_kg * targets['Min_TDN'], "Min TDN"
    prob += sum([INGREDIENT_DATA[i]['TDN'] * x[i] for i in ING_KEYS]) <= target_batch_kg * targets['Max_TDN'], "Max TDN"
    
    # c) ADF (Fiber) constraint (Max)
    prob += sum([INGREDIENT_DATA[i]['ADF'] * x[i] for i in ING_KEYS]) <= target_batch_kg * targets['Max_ADF'], "Max ADF"
    
    # d) Ingredient Maximum Limits (Max_Ingred)
    for ingred, max_prop in targets.get('Max_Ingred', {}).items():
        prob += x[ingred] <= target_batch_kg * max_prop, f"Max {ingred}"
    
    # 5. Solve the problem
    prob.solve()
    
    # 6. Extract Results
    if prob.status == 1: # 1 means optimal solution found
        results = {
            'Cost': f"${value(prob.objective):.2f}",
            'Total_Weight_Check': value(sum([x[i] for i in ING_KEYS])),
            'Ingredients': [
                (i, f"{value(x[i]):.2f} kg") for i in ING_KEYS if value(x[i]) > 0.001
            ]
        }
    else:
        results = None # No feasible solution
        
    return results

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
        """Renders the feed formulation input page."""
        # Pass the keys for the guidance list in the HTML
        available_targets = list(FORMULATION_TARGETS.keys()) 
        return render_template('formulation.html', available_targets=available_targets, result=None)
    # C:\FlaskAppTest\web_app.py (Inside the formulate function)
    @app.route('/formulate', methods=['GET', 'POST'])
    def formulate():
        if request.method == 'GET':
            from flask import redirect, url_for
            return redirect(url_for('formulation_page'))
        try:
            # Get user's typed input
            user_input = request.form['animal_type'].strip().upper() 
            
            # --- NEW FUZZY MATCHING LOGIC ---
            
            # 1. Define the list of available keys to match against
            available_targets = list(FORMULATION_TARGETS.keys())
            
            # 2. Find the best match using fuzzywuzzy
            # Returns (Best Match String, Score, Key)
            best_match = process.extractOne(
                user_input, 
                available_targets,
                scorer=fuzz.ratio # Use the standard Levenshtein ratio
            )
            
            best_match_key = best_match[0]
            score = best_match[1]
            
            # 3. Set a minimum score threshold (e.g., 70 out of 100)
            if score < 70:
                raise KeyError(f"No close match found for '{user_input}'. Best guess: {best_match_key} (Score: {score}).")

            # Use the matched key for formulation
            animal_type = best_match_key
            # --- END FUZZY MATCHING LOGIC ---
            
            # --- LP Solver Call ---
            formulation_result = least_cost_formulate(animal_type, target_batch_kg=100)
            # ... (rest of the logic remains the same) ...
            
        except KeyError as e:
            # Handle cases where input is empty or match is too low
            error_message = f"Invalid animal type or low match score: {e}. Please try a more specific name."
            animal_types = list(FORMULATION_TARGETS.keys())
            # Pass the matched key for suggested input
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