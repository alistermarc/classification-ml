
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.evaluation.evaluate import evaluate_model

def run_classical_ml_experiment():
    """
    This function runs the classical machine learning experiment.
    It loads the data, trains a RandomForestClassifier with GridSearchCV, 
    evaluates the model, and saves the best model.
    """
    print("Running Classical ML Experiment (Random Forest)...")

    # Load the curated data
    with open('data/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the model and GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

    # Train the model
    print("Training RandomForestClassifier with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_

    print("Best parameters found: ", grid_search.best_params_)

    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    evaluate_model(y_test, y_pred, y_pred_proba, model_name='RandomForest')

    # Save the best model
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)

    print("Best RandomForest model saved in 'models/' directory.")

if __name__ == '__main__':
    run_classical_ml_experiment()
