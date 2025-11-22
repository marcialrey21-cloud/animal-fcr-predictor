from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.svm import SVR 
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from thefuzz import fuzz
from thefuzz import process
from pulp import LpProblem, LpMinimize, LpVariable, value
import pickle
import os
import sqlite3

DATABASE = 'fcr_data.db' 
MODEL_FILE = 'model.pkl'
FCR_ANIMAL_TYPES = ['PIG', 'CATTLE', 'POULTRY']

# ====================================================================
# 1. GLOBAL DATA DEFINITIONS (MUST BE OUTSIDE create_app() FOR SCOPE)
# ====================================================================

# --- A. EXPANDED FORMULATION TARGETS (Nutrient Constraints) ---
# NOTE: All min/max values are PROPORTIONS (e.g., 0.18 = 18%)
FORMULATION_TARGETS = {
    # ------------------ SWINE (PIGS) - PIC/BAI Standards ------------------
    'GROWER PIG (25-50 kg)': {
        'Min_Protein': 0.16, 'Max_Protein': 0.18,
        'Min_TDN': 0.78, 'Max_TDN': 0.85, 
        'Min_ADF': 0.00, 'Max_ADF': 0.06,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Rice Bran (D1)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Rice Bran (D1)': 0.30}, 
    },
    'FINISHER PIG (75-100 kg)': {
        'Min_Protein': 0.13, 'Max_Protein': 0.15,
        'Min_TDN': 0.82, 'Max_TDN': 0.90,
        'Min_ADF': 0.00, 'Max_ADF': 0.04,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Rice Bran (D1)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Rice Bran (D1)': 0.40}, 
    },

    # ------------------ POULTRY - BAI/NRC Standards ------------------
    'BROILER CHICK (Starter)': {
        'Min_Protein': 0.22, 'Max_Protein': 0.24, 
        'Min_TDN': 0.85, 'Max_TDN': 0.90,
        'Min_ADF': 0.00, 'Max_ADF': 0.03,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Fish Meal (Local)', 'Limestone (Calcium)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Fish Meal (Local)': 0.08}, 
    },
    'LAYING HEN (Production)': {
        'Min_Protein': 0.16, 'Max_Protein': 0.18,
        'Min_TDN': 0.70, 'Max_TDN': 0.75,
        'Min_ADF': 0.03, 'Max_ADF': 0.07,
        'Ingredients': ['Yellow Corn', 'Soybean Meal (44%)', 'Rice Bran (D1)', 'Limestone (Calcium)', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Limestone (Calcium)': 0.10}, 
    },
    
    # ------------------ RUMINANTS (BEEF/DAIRY) ------------------
    'BEEF CATTLE (Finisher)': {
        'Min_Protein': 0.12, 'Max_Protein': 0.14,
        'Min_TDN': 0.70, 'Max_TDN': 0.75,
        'Min_ADF': 0.25, 'Max_ADF': 0.35, 
        'Ingredients': ['Corn Silage', 'Soybean Meal (44%)', 'Yellow Corn', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Corn Silage': 0.70}, 
    },
    'DAIRY COW (Lactating)': {
        'Min_Protein': 0.17, 'Max_Protein': 0.19,
        'Min_TDN': 0.75, 'Max_TDN': 0.80, 
        'Min_ADF': 0.18, 'Max_ADF': 0.21,
        'Ingredients': ['Corn Silage', 'Alfalfa Hay', 'Soybean Meal (44%)', 'Yellow Corn', 'DCP/Lysine Premix'],
        'Max_Ingred': {'Corn Silage': 0.60, 'Yellow Corn': 0.40}, 
    },
}

# --- B. INGREDIENT DATA (Costs and Nutrient Profiles) ---
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


# ====================================================================
# 2. HELPER FUNCTIONS (Database, Analysis, and Solver)
# ====================================================================

def init_db(app):
    """Initializes the database and creates the prediction table."""
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

def get_analysis_data(app):
    """Fetches all predictions and calculates summary statistics."""
    db_path = os.path.join(app.root_path, DATABASE)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT weight, temperature, predicted_fcr, timestamp FROM predictions ORDER BY timestamp DESC")
    all_predictions = cursor.fetchall()
    
    conn.close()
    
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


def least_cost_formulate(animal_type, target_batch_kg=100, custom_targets=None):
    """Solves for the least-cost feed mix meeting nutritional targets."""
    
    # 1. Determine which targets to use: custom or predefined
    if custom_targets:
        targets = custom_targets
    else:
        targets = FORMULATION_TARGETS.get(animal_type)
    
    if targets is None:
        return None 
    
    ING_KEYS = targets['Ingredients'] 
    
    # 2. Setup the Linear Problem
    prob = LpProblem("Least Cost Feed Mix", LpMinimize)
    x = LpVariable.dicts("Quantity", ING_KEYS, 0)
    
    # 3. Objective Function: Minimize Total Cost
    prob += sum([INGREDIENT_DATA[i]['Cost_USD_kg'] * x[i] for i in ING_KEYS]), "Total Cost"
    
    # 4. Constraints
    for ingredient in ING_KEYS:
        if ingredient == 'DCP/Lysine Premix':
            prob += x[ingredient] >= target_batch_kg * 0.01, f"Min {ingredient} (Quality)"
        elif ingredient == 'Rice Bran (D1)':
            prob += x[ingredient] >= target_batch_kg * 0.10, f"Min {ingredient} (Staple)"
        prob += x[ingredient] <= target_batch_kg * 0.70, f"Max {ingredient} (Diversity)"
    prob += sum([x[i] for i in ING_KEYS]) == target_batch_kg, "Total Weight"

    # Protein Constraints
    
    prob += sum([INGREDIENT_DATA[i]['Protein'] * x[i] for i in ING_KEYS]) >= target_batch_kg * targets['Min_Protein'], "Min Protein"
    prob += sum([INGREDIENT_DATA[i]['Protein'] * x[i] for i in ING_KEYS]) <= target_batch_kg * targets['Max_Protein'], "Max Protein"

    # TDN (Energy) Constraints
    prob += sum([INGREDIENT_DATA[i]['TDN'] * x[i] for i in ING_KEYS]) >= target_batch_kg * targets['Min_TDN'], "Min TDN"
    prob += sum([INGREDIENT_DATA[i]['TDN'] * x[i] for i in ING_KEYS]) <= target_batch_kg * targets['Max_TDN'], "Max TDN"
    
    # ADF (Fiber) Constraint
    prob += sum([INGREDIENT_DATA[i]['ADF'] * x[i] for i in ING_KEYS]) <= target_batch_kg * targets['Max_ADF'], "Max ADF"
    
    # Ingredient Maximum Limits
    for ingred, max_prop in targets.get('Max_Ingred', {}).items():
        prob += x[ingred] <= target_batch_kg * max_prop, f"Max {ingred}"
    
    # 5. Solve the problem
    prob.solve()
    
    # 6. Extract Results
    if prob.status == 1: 
        return {
            'Status': 'Optimal Solution Found',
            'Cost': f"${value(prob.objective):.2f}",
            'Total_Weight_Check': value(sum([x[i] for i in ING_KEYS])),
            'Ingredients': [
                (i, f"{value(x[i]):.2f} kg") for i in ING_KEYS if value(x[i]) > 0.001
            ]
        }
    else:
        return {'Status': 'No Feasible Solution Found'}


# ====================================================================
# 3. APPLICATION FACTORY AND ROUTES
# ====================================================================

def create_app():
    app = Flask(__name__)
    
    # --- Model Loading / Training Logic ---
    # 1. EXPANDED TRAINING DATA (Simulated data for three species)
    raw_data = {
        'Type': ['PIG'] * 5 + ['CATTLE'] * 5 + ['POULTRY'] * 5,
        'Weight_kg': [50, 55, 60, 45, 70,    200, 250, 300, 150, 350,   1.5, 2.0, 1.2, 1.8, 2.5],
        'Temp_C': [20, 21, 22, 19, 23,        15, 18, 16, 20, 17,       25, 24, 26, 23, 27],
        'FCR': [2.5, 2.4, 2.6, 2.3, 2.7,      6.5, 6.4, 6.8, 6.0, 7.0,   1.8, 1.7, 1.9, 1.6, 2.0]
    }

    df = pd.DataFrame(raw_data)
    model_path = os.path.join(app.root_path, MODEL_FILE)

    if os.path.exists(model_path):
        print(f"Loading trained model and scalers from {MODEL_FILE}")
        with open(model_path, 'rb') as file:
            saved_objects = pickle.load(file) 
            model = saved_objects['model']
            X_scaler = saved_objects['X_scaler']
            y_scaler = saved_objects['y_scaler']
            ohe = saved_objects['ohe']
    else:
        print("Model file not found. Starting training and hyperparameter tuning...")
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat_encoded = ohe.fit_transform(df[['Type']])

        X_num = df[['Weight_kg', 'Temp_C']].values
        X_scaler = StandardScaler() 
        X_num_scaled = X_scaler.fit_transform(X_num)

        X_final_scaled = np.hstack((X_num_scaled, X_cat_encoded))

        y_reshaped = df['FCR'].values.reshape(-1, 1)
        y_scaler = StandardScaler() 
        y_scaled = y_scaler.fit_transform(y_reshaped)

        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
        grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
        grid_search.fit(X_final_scaled, y_scaled.ravel())
        
        model = grid_search.best_estimator_
        print(f"Tuning complete. Best parameters found: {grid_search.best_params_}")

        saved_objects = {'model': model, 'X_scaler': X_scaler, 'y_scaler': y_scaler, 'ohe': ohe}
        with open(model_path, 'wb') as file:
            pickle.dump(saved_objects, file)
        print(f"Model and scalers saved to {MODEL_FILE}")


    # --- Prediction Function (Updated to accept animal_type) ---
    def predict_fcr(animal_type, weight, temperature):

        new_data_num = np.array([[weight, temperature]])
        input_df = pd.DataFrame([{'Type': animal_type}])

        new_data_cat_encoded = ohe.transform(input_df[['Type']])

        new_data_num_scaled = X_scaler.transform(new_data_num)

        new_data_final = np.hstack((new_data_num_scaled, new_data_cat_encoded))
        predicted_fcr_scaled = model.predict(new_data_final)

        predicted_fcr_reshaped = predicted_fcr_scaled.reshape(-1, 1)
        predicted_fcr = y_scaler.inverse_transform(predicted_fcr_reshaped)[0][0]
        return predicted_fcr


    # --- Database Initialization (Runs once per startup) ---
    with app.app_context():
        init_db(app)    

    # ====================================================================
    # 4. FLASK ROUTES
    # ====================================================================
    
    @app.route('/')
    def home():
        animal_types_list = FCR_ANIMAL_TYPES

        if not animal_types_list:
            print("CRITICAL: FCR_ANIMAL_TYPES is empty at runtime.")
        
        return render_template('index.html', result=None, fcr_animal_types=FCR_ANIMAL_TYPES)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            animal_type = request.form['animal_type'] # NEW: Get Animal Type
            weight_str = request.form['weight']
            temp_str = request.form['temp']
            if not weight_str or not temp_str: raise KeyError 

            weight = float(weight_str)
            temp = float(temp_str)
        
        except KeyError:
            error_message = "Please fill in ALL fields."
            return render_template('index.html', result=None, error=error_message, fcr_animal_types=FCR_ANIMAL_TYPES)

        except ValueError:
            error_message = "Please enter valid NUMERIC values."
            return render_template('index.html', result=None, error=error_message, fcr_animal_types=FCR_ANIMAL_TYPES)

        # Prediction & Database Saving
        predicted_value = predict_fcr(animal_type, weight, temp)

        recommendation = "FCR is high. Consider adjusting diet composition or checking for heat stress." if predicted_value > 2.6 else "FCR is within an acceptable range for current conditions."
        result_data = {'animal_type': animal_type, 'weight': weight, 'temp': temp, 'fcr': f'{predicted_value:.3f}', 'recommendation': recommendation}
        
        return render_template('index.html', result=result_data, fcr_animal_types=FCR_ANIMAL_TYPES)

    @app.route('/formulation')
    def formulation_page():
        """Renders the standard feed formulation input page."""
        available_targets = list(FORMULATION_TARGETS.keys()) 
        return render_template('formulation.html', available_targets=available_targets, result=None)
    
    @app.route('/calculator')
    def calculator_page():
        """Renders the custom multi-ingredient calculator input form."""
        ingredient_list = list(INGREDIENT_DATA.keys()) 
        return render_template('calculator.html', ingredient_list=ingredient_list, result=None)

    @app.route('/calculate_mix', methods=['POST'])
    def calculate_mix():
        # This route should be safe now that INGREDIENT_DATA is global
        ingredient_list = list(INGREDIENT_DATA.keys())
        
        try:
            target_cp = float(request.form['cp_target'])
            
            # Get the list of ingredients the user selected
            selected_ingredients = [
                request.form[key] for key in request.form 
                if key.startswith('ingredient_') and request.form[key] in INGREDIENT_DATA
            ]
            
            if len(selected_ingredients) < 2:
                return render_template('calculator.html', ingredient_list=ingredient_list, error="Please select at least two ingredients for formulation.")

            if target_cp <= 0 or target_cp > 100:
                raise ValueError("Target CP must be a value between 1 and 100.")

            # Create custom targets for the solver
            custom_targets = {
                'Min_Protein': target_cp / 100.0, 
                'Max_Protein': (target_cp + 2) / 100.0, 
                'Min_TDN': 0.70, 'Max_TDN': 0.95, 
                'Min_ADF': 0.00, 'Max_ADF': 0.10, 
                'Ingredients': selected_ingredients,
                'Max_Ingred': {} 
            }
            
            # Run the LP Solver
            results = least_cost_formulate('CUSTOM_MIX', target_batch_kg=100, custom_targets=custom_targets)
            
            if results is None or results.get('Status') == 'No Feasible Solution Found':
                return render_template('calculator.html', ingredient_list=ingredient_list, error="Optimization failed. Ingredients cannot meet the target CP or constraints. Try a lower CP or adding protein/energy sources.")

            # Success: Format results
            result_data = {'cp_target': target_cp, 'lp_results': results}
            
            return render_template('calculator.html', ingredient_list=ingredient_list, result=result_data)
            
        except ValueError as e:
            return render_template('calculator.html', ingredient_list=ingredient_list, error=f"Invalid Input: {e}")
        except Exception as e:
            return render_template('calculator.html', ingredient_list=ingredient_list, error=f"An unexpected server error occurred: {e}")


    @app.route('/formulate', methods=['GET', 'POST'])
    def formulate():
        if request.method == 'GET':
            return redirect(url_for('formulation_page'))
        
        # --- Handle POST Request (Form Submission) ---
        try:
            # Get user's typed input
            user_input = request.form['animal_type'].strip().upper() 
            
            available_targets = list(FORMULATION_TARGETS.keys())
            
            # Fuzzy match to find the correct profile
            best_match = process.extractOne(user_input, available_targets, scorer=fuzz.ratio)
            best_match_key = best_match[0]
            score = best_match[1]
            
            if score < 70:
                raise KeyError(f"No close match found for '{user_input}'. Best guess: {best_match_key} (Score: {score}).")

            animal_type = best_match_key
            
            # LP Solver Call
            formulation_result = least_cost_formulate(animal_type, target_batch_kg=100)

            # Check for solver failure
            if formulation_result is None or formulation_result.get('Status') == 'No Feasible Solution Found':
                raise Exception("Optimization failed due to strict constraints.")
            
            
            # Prepare successful result data
            result_data = {'type': animal_type, 'targets': FORMULATION_TARGETS[animal_type], 'lp_results': formulation_result}

            available_targets = list(FORMULATION_TARGETS.keys())
            return render_template('formulation.html', available_targets=available_targets, result=result_data)
            
        except KeyError as e:
            error_message = f"Input Error: {e}"
            available_targets = list(FORMULATION_TARGETS.keys())
            return render_template('formulation.html', available_targets=available_targets, error=error_message)
        
        except Exception as e:
            error_message = f"Optimization Failed: {e}. Constraints may be impossible to meet."
            available_targets = list(FORMULATION_TARGETS.keys())
            return render_template('formulation.html', available_targets=available_targets, error=error_message)


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


# --- Local Runner for Development ONLY ---
if __name__ == '__main__':
    local_app = create_app()
    local_app.run(debug=True)