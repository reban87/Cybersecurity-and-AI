import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load and prepare the data
def load_and_prepare_data():
    # Load datasets
    train_data = pd.read_csv('src/dataset/network_traffic_training.csv')
    test_data = pd.read_csv('src/dataset/network_traffic_test.csv')
    
    # Convert timestamp to datetime
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    
    return train_data, test_data


# 2. Feature Engineering
def engineer_features(df):
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Create network-based features
    df['bytes_per_packet'] = df['bytes'] / df['packets']
    df['packet_rate'] = df['packets'] / df['duration']
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['protocol_encoded'] = le.fit_transform(df['protocol'])
    
    # Extract IP features
    df['source_ip_first_octet'] = df['source_ip'].apply(lambda x: int(x.split('.')[0]))
    df['dest_ip_first_octet'] = df['dest_ip'].apply(lambda x: int(x.split('.')[0]))
    
    return df


# 3. Prepare features for modeling
def prepare_features(df):
    feature_columns = [
        'bytes', 'packets', 'duration', 'bytes_per_packet', 'packet_rate',
        'protocol_encoded', 'port', 'hour', 'day_of_week',
        'source_ip_first_octet', 'dest_ip_first_octet'
    ]
    
    X = df[feature_columns]
    y = df['is_attack']
    
    return X, y


# 4. Model training and evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, y_pred, y_pred_prob, X_train_scaled, X_test_scaled


# 5. Visualization functions
def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance(model, feature_columns):
    importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Feature Importance')
    plt.show()


# 6. Main execution
def main():
    # Load and prepare data
    print("Loading and preparing data...")
    train_data, test_data = load_and_prepare_data()
    
    # Engineer features
    print("Engineering features...")
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)
    
    # Prepare features
    print("Preparing features for modeling...")
    X_train, y_train = prepare_features(train_data)
    X_test, y_test = prepare_features(test_data)
    
    # Train and evaluate model
    print("Training and evaluating model...")
    model, y_pred, y_pred_prob, X_train_scaled, X_test_scaled = train_and_evaluate_model(
        X_train, X_test, y_train, y_test
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)
    plot_feature_importance(model, X_train.columns)
    
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = main()