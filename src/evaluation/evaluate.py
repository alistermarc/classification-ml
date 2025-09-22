from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_pred, y_pred_proba, model_name='model'):
    """
    This function evaluates a classification model and prints the results.
    """
    print(f'--- Evaluation for {model_name} ---')
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    
    print(f'ROC-AUC Score: {roc_auc:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Print and plot confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Create a directory for evaluation plots
    eval_path = 'evaluation_plots'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
        
    plt.savefig(os.path.join(eval_path, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    print(f"Confusion matrix plot saved in '{eval_path}' directory.")
    print('-------------------------------------\n')
    
    return {'roc_auc': roc_auc, 'f1_score': f1}


