import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin



def preprocess_bank_loan_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Handle missing values
    # Identifying numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Identifying categorical columns (explicitly defining)
    cat_cols = ['ed']  # 'ed' is categorical, encoded from 1 to 5

    # Impute missing values for numerical columns with the median
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute missing values for categorical columns with the mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode the categorical variable 'ed' using LabelEncoder
    label_encoder = LabelEncoder()
    df['ed'] = label_encoder.fit_transform(df['ed'])

    # # Visualize the distribution of the target variable
    # plt.figure(figsize=(6, 4))
    # sns.countplot(x='default', data=df, palette='coolwarm')
    # plt.title('Distribution of Default Status')
    # plt.xlabel('Default')
    # plt.ylabel('Count')
    # plt.show()

    # # Visualize the correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    # plt.title('Correlation Heatmap')
    # plt.show()

    # # Visualize pairwise relationships
    # sns.pairplot(df, hue='default', diag_kind='kde', palette='coolwarm')
    # plt.show()

    return df



def split_and_scale_data(df):
    # Splitting the dataset into features (X) and target (y)
    X = df.drop(columns=['default'])
    y = df['default']

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Display the scaled data
    print("\nScaled Data (First 5 rows of X_train_scaled):")
    print(X_train_scaled[:5])

    print("\nPreprocessing Complete. Data is ready for modeling.")
    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy()




def evaluate_random_forest_model(X_train, y_train, n_estimators=50, max_depth=None, min_samples_split=5):
    # Initialize the Random Forest model with provided hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        random_state=42
    )
    
    # KFold Cross-validation setup
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_per_fold = []
    y_true_all, y_proba_all = [], []

    # Perform 10-Fold Cross-validation
    for fold_number, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train), 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Fit model on the fold
        rf_model.fit(X_train_fold, y_train_fold)
        y_pred = rf_model.predict(X_val_fold)
        y_proba = rf_model.predict_proba(X_val_fold)[:, 1]

        # Append true labels and predicted probabilities for ROC calculation
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

        # Calculate metrics
        tp = np.sum((y_val_fold == 1) & (y_pred == 1))
        tn = np.sum((y_val_fold == 0) & (y_pred == 0))
        fp = np.sum((y_val_fold == 0) & (y_pred == 1))
        fn = np.sum((y_val_fold == 1) & (y_pred == 0))
        
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * precision * true_positive_rate / (precision + true_positive_rate) if (precision + true_positive_rate) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        error_rate = 1 - accuracy
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        true_skill_statistic = true_positive_rate + true_negative_rate - 1
        heidke_skill_score = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
        brier_score = brier_score_loss(y_val_fold, y_proba)
        auc_score = roc_auc_score(y_val_fold, y_proba)

        # Append metrics for each fold
        metrics_per_fold.append([
            fold_number, tp, tn, fp, fn, true_positive_rate, true_negative_rate, 
            false_positive_rate, false_negative_rate, precision, f1_score, 
            accuracy, error_rate, balanced_accuracy, true_skill_statistic, 
            heidke_skill_score, brier_score, auc_score
        ])

    # Create DataFrame with fold metrics
    metrics_rf = pd.DataFrame(metrics_per_fold, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
        'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS', 
        'Brier_score', 'AUC'
    ])

    # Calculate average metrics across all folds
    metrics_rf.loc['Average'] = metrics_rf.mean(numeric_only=True)

    # Calculate overall ROC AUC and plot ROC curve
    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    fpr, tpr, thresholds = roc_curve(y_true_all, y_proba_all)
    roc_auc = roc_auc_score(y_true_all, y_proba_all)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()

    return metrics_rf, roc_auc



def evaluate_svm_model(X_train, y_train):
    # Best hyperparameters from previous tuning
    best_svm = SVC(
        C=1, kernel='linear', gamma='scale', probability=True, random_state=42
    )
    
    # KFold Cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []
    y_true_all = []
    y_proba_all = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Fit the model on the fold
        best_svm.fit(X_train_fold, y_train_fold)
        y_pred = best_svm.predict(X_val_fold)
        y_proba = best_svm.predict_proba(X_val_fold)[:, 1]
        
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

        # Initialize counts
        tp = tn = fp = fn = 0

        # Calculate manually
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1

        # Calculate metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Error_rate = 1 - Accuracy
        BACC = (TPR + TNR) / 2
        TSS = TPR + TNR - 1
        HSS = 2 * (tp * tn - fp * fn) / (
            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        ) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)

        # Append metrics for each fold
        metrics_list.append(
            [
                fold, tp, tn, fp, fn, TPR, TNR, FPR, FNR, Precision, F1,
                Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC
            ]
        )

    # Create DataFrame with fold metrics
    metrics_svm = pd.DataFrame(
        metrics_list, columns=[
            'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
            'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC',
            'TSS', 'HSS', 'Brier_score', 'AUC'
        ]
    )

    # Calculate averages and append to the DataFrame
    metrics_svm.loc['Average'] = metrics_svm.mean(numeric_only=True)

    # Aggregate true labels and predicted probabilities for ROC curve
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc = roc_auc_score(y_true_all, y_proba_all)

    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for SVM Model (10-Fold CV)')
    plt.legend()
    plt.show()

    # Return metrics DataFrame and ROC information
    return metrics_svm, roc_auc




def evaluate_decision_tree_model(X_train, y_train):
    # Best Decision Tree model after hyperparameter tuning
    best_dt = DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=10, min_samples_leaf=5)

    # KFold Cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []
    y_true_all = []
    y_proba_all = []

    # Perform 10-Fold Cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Fit model on the fold
        best_dt.fit(X_train_fold, y_train_fold)
        y_pred = best_dt.predict(X_val_fold)
        y_proba = best_dt.predict_proba(X_val_fold)[:, 1]

        # Append true labels and probabilities for ROC calculation
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

        # Initialize counts
        tp = tn = fp = fn = 0

        # Loop through true and predicted labels
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1  # True Positive
            elif true == 0 and pred == 0:
                tn += 1  # True Negative
            elif true == 0 and pred == 1:
                fp += 1  # False Positive
            elif true == 1 and pred == 0:
                fn += 1  # False Negative

        # Calculate metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Error_rate = 1 - Accuracy
        BACC = (TPR + TNR) / 2
        TSS = TPR + TNR - 1
        HSS = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)

        # Append metrics for each fold
        metrics_list.append([fold, tp, tn, fp, fn, TPR, TNR, FPR, FNR, Precision, F1, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

    # Create DataFrame with fold metrics
    metrics_dt = pd.DataFrame(metrics_list, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
        'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC',
        'TSS', 'HSS', 'Brier_score', 'AUC'
    ])

    # Calculate average metrics across all folds
    metrics_dt.loc['Average'] = metrics_dt.mean(numeric_only=True)

    # Calculate ROC curve and AUC across all folds
    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc = roc_auc_score(y_true_all, y_proba_all)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()

    print(f"Average ROC AUC Score across folds: {roc_auc:.2f}")

    return metrics_dt, roc_auc




# Function to evaluate the LSTM model and collect metrics
def evaluate_lstm_model(df):
    # Separate features and target
    X = df.drop('default', axis=1).values
    y = df['default'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape data for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Define custom Keras model wrapper
    class KerasLSTMClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, learning_rate=0.001, dropout_rate=0.2, epochs=20, batch_size=32):
            self.learning_rate = learning_rate
            self.dropout_rate = dropout_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None

        def fit(self, X, y):
            self.model = self.create_lstm_model()
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self

        def predict(self, X):
            return (self.model.predict(X) > 0.5).astype("int32").flatten()

        def create_lstm_model(self):
            model = Sequential()
            model.add(LSTM(units=50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(1, activation='sigmoid'))
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return model

    # Hyperparameters from the previous tuning
    best_params = {'dropout_rate': 0.3, 'learning_rate': 0.01}
    
    # Set up 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize model with best hyperparameters
    model = KerasLSTMClassifier(
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout_rate']
    )

    # Create a list to store metrics for each fold
    metrics_list = []
    y_true_all = []
    y_proba_all = []

    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train the model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on the validation fold
        y_pred = model.predict(X_val_fold)
        y_proba = model.model.predict(X_val_fold).flatten()

        # Collect true labels and predicted probabilities
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

        # Initialize counts for confusion matrix
        tp = tn = fp = fn = 0
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1  # True Positive
            elif true == 0 and pred == 0:
                tn += 1  # True Negative
            elif true == 0 and pred == 1:
                fp += 1  # False Positive
            elif true == 1 and pred == 0:
                fn += 1  # False Negative
        
        # Calculate metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Error_rate = 1 - Accuracy
        BACC = (TPR + TNR) / 2
        TSS = TPR + TNR - 1
        HSS = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)

        # Append metrics for each fold
        metrics_list.append([fold, tp, tn, fp, fn, TPR, TNR, FPR, FNR, Precision, F1, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

    # Create DataFrame with fold metrics
    metrics_lstm = pd.DataFrame(metrics_list, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
        'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC',
        'TSS', 'HSS', 'Brier_score', 'AUC'
    ])

    # Calculate and append the average of metrics
    metrics_lstm.loc['Average'] = metrics_lstm.mean(numeric_only=True)

    # Compute average ROC AUC
    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    roc_auc_avg = roc_auc_score(y_true_all, y_proba_all)
     
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true_all, y_proba_all)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_avg:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LSTM Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()

    return metrics_lstm, roc_auc_avg


def merged_metrics(metrics_rf, metrics_svm, metrics_dt, metrics_lstm):

    # Add a new column to each DataFrame for the model name
    metrics_rf['Model'] = 'Random Forest'
    metrics_svm['Model'] = 'SVM'
    metrics_dt['Model'] = 'Decision Tree'
    metrics_lstm['Model'] = 'LSTM'

    # Select the last row from each DataFrame and merge them vertically
    merged_metrics = pd.concat([metrics_rf[-1:], metrics_svm[-1:], metrics_dt[-1:], metrics_lstm[-1:]], ignore_index=True)
    
    # Move the 'Model' column to the first position
    merged_metrics = merged_metrics[['Model'] + [col for col in merged_metrics.columns if col != 'Model']]

    # Transpose the DataFrame to have models as column names
    merged_metrics = merged_metrics.set_index('Model').T

    # Display the DataFrame with bold model names in the first row
    # Formatting the model names as bold for display (works in Jupyter environments)
    merged_metrics.columns = [f"{col}" for col in merged_metrics.columns]

    # Display the transposed DataFrame
    print(merged_metrics)

    return merged_metrics


def best_model_result(merged_metrics):
    # Assuming merged_metrics is the transposed DataFrame with model metrics
    # Select key metrics for comparison
    key_metrics = ['Accuracy', 'AUC', 'Precision', 'F1_measure', 'BACC', 'HSS']

    # Filter the merged metrics DataFrame to only key metrics
    metrics_for_ranking = merged_metrics.loc[key_metrics]

    # Rank each model for each metric (higher is better, so we rank by descending values)
    ranks = metrics_for_ranking.rank(ascending=False, axis=1)

    # Sum ranks for each model to get a total score (lower score indicates better performance)
    total_scores = ranks.sum()

    # Find the model with the lowest total score
    best_model = total_scores.idxmin()

    # Display the ranking results and the best model
    print("\nRanking of Models by Metrics:\n", ranks)
    print("\nTotal Scores for Each Model:\n", total_scores)
    print(f"\nBest Model Overall: {best_model}")



def main():
    # Path to the dataset
    file_path = 'Bankloan.csv'

    # Step 1: Preprocess the dataset
    preprocessed_df = preprocess_bank_loan_data(file_path)

    # Step 2: Split and scale the dataset
    X_train, X_test, y_train, y_test = split_and_scale_data(preprocessed_df)

    # You can now use X_train_scaled, X_test_scaled, y_train, and y_test for model training and evaluation
    print("\nData is prepared for model training and testing.")

    # Random Forest Classifier
    metrics_rf, roc_auc_rf = evaluate_random_forest_model(X_train, y_train)
    print("\nRandom Forest Model Evaluation Metrics:\n")
    print(metrics_rf)
    print(f"Average ROC AUC Score: {roc_auc_rf:.2f}")

    # SVM
    metrics_svm, roc_auc_svm = evaluate_svm_model(X_train, y_train)
    print("\nSVM Model Evaluation Metrics:\n")
    print(metrics_svm)
    print(f"Average ROC AUC Score: {roc_auc_svm:.2f}")

    # DecisionTreeClassifier
    metrics_dt, roc_auc_dt = evaluate_decision_tree_model(X_train, y_train)
    print("\nDecision Tree Model Evaluation Metrics:\n")
    print(metrics_dt)
    print(f"Average ROC AUC Score: {roc_auc_dt:.2f}")

    # LSTM
    metrics_lstm, roc_auc_lstm = evaluate_lstm_model(preprocessed_df)
    print("\nLSTM Model Evaluation Metrics:\n")
    print(metrics_lstm)
    print(f"Average ROC AUC Score: {roc_auc_lstm:.2f}")

    #Merging all the metrics to a single table (Only the average metrics)
    merged_metric = merged_metrics(metrics_rf, metrics_svm, metrics_dt, metrics_lstm)

    #Result 
    print("\n")
    best_model_result(merged_metric)

if __name__ == "__main__":
    main()
