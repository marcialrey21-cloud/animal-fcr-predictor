from flask import Flask, request, render_template 
import numpy as np
from sklearn.svm import SVR 
import pickle
import os
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# Note: Ensure you have flask, numpy, scikit-learn installed: 
# pip install flask scikit-learn numpy

# --- 1. Model Setup Data ---

# Define the filename for saving the model and scalers
MODEL_FILE = 'model.pkl'

# Features (X) and Target (y) data for training
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

# --- Feed Formulation Data (Nutritional Targets) ---
FORMULATION_TARGETS = {
    'CALF': {
        'Protein': '18-22%',
        'Energy (TDN)': '75%',
        'Fiber (ADF)': '15-20%',
        'Ingredients': ['Forage (Hay/Silage)', 'Grain Mix (Corn/Soybean Meal)', 'Mineral/Vitamin Premix']
    },
    'FINISHING PIG': {
        'Protein': '14-16%',
        'Energy (TDN)': '85%',
        'Fiber (ADF)': '<5%',
        'Ingredients': ['Corn/Barley', 'Soybean Meal', 'Fat/Oil Source', 'Mineral/Vitamin Premix']
    },
    'LAYING HEN': {
        'Protein': '16-18%',
        'Energy (TDN)': '60%',
        'Fiber (ADF)': '3-5%',
        'Ingredients': ['Corn/Wheat', 'Soybean Meal', 'Calcium Source (Limestone)', 'Mineral/Vitamin Premix']
    }
}


# --- Model Loading / Training Logic ---
if os.path.exists(MODEL_FILE):
    print(f"Loading trained model and scalers from {MODEL_FILE}")
    with open(MODEL_FILE, 'rb') as file:
        saved_objects = pickle.load(file) 
        model = saved_objects['model']
        X_scaler = saved_objects['X_scaler']
        y_scaler = saved_objects['y_scaler']
else:
    print("Model file not found. Starting training and hyperparameter tuning...")
    
    # Initialize the scalers
    X_scaler = StandardScaler() 
    y_scaler = StandardScaler() 
    
    # Scale the data
    X_scaled = X_scaler.fit_transform(X)
    y_reshaped = y.reshape(-1, 1)
    y_scaled = y_scaler.fit_transform(y_reshaped)

    # Define the parameter grid for SVR tuning
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000], 
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    # Setup and run Grid Search
    grid_search = GridSearchCV(
        estimator=SVR(),         
        param_grid=param_grid,   
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y_scaled.ravel())
    
    # Select the best model
    model = grid_search.best_estimator_
    print(f"Tuning complete. Best parameters found: {grid_search.best_params_}")
    
    # Save the best model and scalers
    saved_objects = {
        'model': model,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(saved_objects, file)
    print(f"Best model and scalers saved to {MODEL_FILE}")

DATABASE = 'fcr_data.db' 

def init_db():
    """Initializes the database and creates the prediction table."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist. We save the inputs and the final prediction.
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

# Call the initialization function immediately after defining it
init_db()

# --- Prediction Function ---
def predict_fcr(weight, temperature):
    """Predicts the FCR after scaling input and inverse scaling output."""
    new_data = np.array([[weight, temperature]])
    
    # Scale the new input data
    new_data_scaled = X_scaler.transform(new_data)
    
    # Run the prediction
    predicted_fcr_scaled = model.predict(new_data_scaled)
    
    # Inverse scale the prediction to get the real FCR value
    predicted_fcr_reshaped = predicted_fcr_scaled.reshape(-1, 1)
    predicted_fcr = y_scaler.inverse_transform(predicted_fcr_reshaped)[0][0]
    
    return predicted_fcr


# --- 2. Flask Application Routes ---

# **CRUCIAL: INITIALIZE APP BEFORE ROUTES**
app = Flask(__name__) 

# ... (After predict_fcr function) ...

def get_analysis_data():
    """Fetches all predictions and calculates summary statistics."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Fetch all data from the predictions table
    cursor.execute("SELECT weight, temperature, predicted_fcr, timestamp FROM predictions ORDER BY timestamp DESC")
    all_predictions = cursor.fetchall()
    
    conn.close()
    
    data_list = [{'weight': w, 'temp': t, 'fcr': f, 'time': ts} 
                 for w, t, f, ts in all_predictions]
    
    # Calculate summary statistics using numpy if data exists
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

# Route for the Home Page (Input Form)
@app.route('/')
def home():
    """Renders the main prediction input page."""
    return render_template('index.html', result=None)

# Route to Handle the Form Submission (Prediction)
@app.route('/predict', methods=['POST'])
def predict():
    """Handles input validation and returns the FCR prediction."""
    try:
        # Input Validation Check 1: Missing or Empty Fields (KeyError)
        weight_str = request.form['weight']
        temp_str = request.form['temp']
        
        if not weight_str or not temp_str:
             # Raise KeyError if fields are explicitly empty
             raise KeyError 

        # Input Validation Check 2: Non-numeric Data (ValueError)
        weight = float(weight_str)
        temp = float(temp_str)
    
    except KeyError:
        error_message = "Please fill in BOTH Animal Weight and Ambient Temperature."
        return render_template('index.html', result=None, error=error_message)

    except ValueError:
        error_message = "Please enter valid NUMERIC values for Weight and Temperature."
        return render_template('index.html', result=None, error=error_message)

    # --- Prediction Logic ---
    predicted_value = predict_fcr(weight, temp)
    
    # Create a recommendation message
    if predicted_value > 2.6:
        recommendation = "FCR is high. Consider adjusting diet composition or checking for heat stress."
    else:
        recommendation = "FCR is within an acceptable range for current conditions."
        
    result_data = {
        'weight': weight,
        'temp': temp,
        'fcr': f'{predicted_value:.3f}',
        'recommendation': recommendation
    }

    return render_template('index.html', result=result_data)


# Route for the Feed Formulation Page
@app.route('/formulation')
def formulation_page():
    """Renders the feed formulation input page."""
    animal_types = list(FORMULATION_TARGETS.keys())
    return render_template('formulation.html', animal_types=animal_types, result=None)


# Route to handle the selection of the animal type
@app.route('/formulate', methods=['POST'])
def formulate():
    """Handles the animal type selection and returns the feed formulation targets."""
    try:
        animal_type = request.form['animal_type']
        
        targets = FORMULATION_TARGETS.get(animal_type, None)
        
        if targets:
            result_data = {
                'type': animal_type,
                'targets': targets
            }
        else:
            # Re-render with an error if the type isn't found
            return formulation_page(error="Invalid animal type selected.")

        animal_types = list(FORMULATION_TARGETS.keys())
        return render_template('formulation.html', animal_types=animal_types, result=result_data)

    except KeyError:
        error_message = "Please select an animal type."
        animal_types = list(FORMULATION_TARGETS.keys())
        return render_template('formulation.html', animal_types=animal_types, error=error_message)

# Route for the Data Analysis Page
@app.route('/analysis')
def data_analysis():
    """Renders the data analysis page with prediction history and stats."""
    
    summary, all_predictions = get_analysis_data()
    
    # Use f-string formatting for cleaner display in the template
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

# --- 3. Run the Server ---
if __name__ == '__main__':
    # Start the server
    app.run(debug=True)