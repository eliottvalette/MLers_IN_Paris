# %% [markdown]
# # Importing Libraries and Datasets

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
import warnings

warnings.filterwarnings("ignore")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    fxn()

# %%
# Load the training dataset
instagram_df_train=pd.read_csv('kaggle/input/Dataset_1/train.csv')
instagram_df_train

# %%
# Load the testing data
instagram_df_test=pd.read_csv('kaggle/input/Dataset_1/test.csv')
instagram_df_test

# %% [markdown]
# # Statistical Analysis

# %%
instagram_df_train.head()

# %%
instagram_df_train.tail()

# %%
# Getting dataframe info
instagram_df_train.info()

# %%
# Get the statistical summary of the dataframe
instagram_df_train.describe()

# %%
# Get the number of unique values in the "profile pic" feature
instagram_df_train['profile pic'].value_counts()

# %%
# Get the number of unique values in "fake" (Target column)
instagram_df_train['fake'].value_counts()

# %% [markdown]
# # Data Visualization

# %%
# Visualization of real vs fake profiles distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='fake', data=instagram_df_train, palette=['#3498db', '#e74c3c'])
plt.title('Instagram Profile Distribution: Real vs Fake', fontsize=16)
plt.xlabel('Fake Profile (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Number of Profiles', fontsize=12)
# Add values on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom', fontsize=12)

plt.xticks([0, 1], ['Real (0)', 'Fake (1)'])
plt.show()

# %%
# Visualization of the digits/length ratio distribution of usernames
plt.figure(figsize = (14, 8))
ax = sns.histplot(instagram_df_train['nums/length username'], bins=30, kde=True, color='#3498db')
plt.title('Digits/Length Ratio Distribution of Usernames', fontsize=16)
plt.xlabel('Digits/Length Ratio of Username', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add a vertical line for the mean
mean_val = instagram_df_train['nums/length username'].mean()
plt.axvline(x=mean_val, color='#e74c3c', linestyle='--', linewidth=2)
plt.text(mean_val + 0.02, plt.ylim()[1]*0.9, f'Mean: {mean_val:.3f}', color='#e74c3c', fontsize=12)

plt.show()

# %%
# Enhanced visualization of the correlation matrix
plt.figure(figsize=(16, 14))
mask = np.triu(instagram_df_train.corr())
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(instagram_df_train.corr(), annot=True, fmt='.2f', cmap=cmap, linewidths=0.5, 
            mask=mask, vmin=-1, vmax=1, center=0, square=True, cbar_kws={"shrink": .8})

plt.title('Feature Correlation Matrix', fontsize=18, pad=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Data Modelling

# %%
# Training and testing dataset (inputs)
X_train = instagram_df_train.drop(columns = ['fake'])
X_test = instagram_df_test.drop(columns = ['fake'])
X_train

# %%
# Training and testing dataset (Outputs)
y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']
y_train

# %%
# Scale the data before training the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# %% [markdown]
# # Neural Network Model

# %%
# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)

# %%
# Convert labels to one-hot encoding
y_train_tensor = torch.zeros(len(y_train), 2)
y_train_tensor[range(len(y_train)), y_train.astype(int)] = 1
y_test_tensor = torch.zeros(len(y_test), 2)
y_test_tensor[range(len(y_test)), y_test.astype(int)] = 1

# %%
# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# %%
# Create dataloaders
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# %%
# PyTorch model definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=11):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 150)
        self.layer3 = nn.Linear(150, 150)
        self.layer4 = nn.Linear(150, 25)
        self.layer5 = nn.Linear(25, 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.softmax(self.layer5(x))
        return x

# %%
# Create the model
model = NeuralNetwork()
print(model)

# %%
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Training the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 50
history = {'train_loss': [], 'val_loss': []}

# %%
# Validation split
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

# %%
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    history['val_loss'].append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')


# %% [markdown]
# # # Neural Network Model Validation and Results

# %%
print(history.keys())

# %%
# Enhanced visualization of loss progression
plt.figure(figsize=(12, 7))
epochs = range(1, len(history['train_loss']) + 1)

plt.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')

plt.title('Training and Validation Loss Evolution', fontsize=16)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

min_val_loss = min(history['val_loss'])
min_val_epoch = history['val_loss'].index(min_val_loss) + 1
plt.plot(min_val_epoch, min_val_loss, 'ro', markersize=8)
plt.annotate(f'Min: {min_val_loss:.4f}', 
             xy=(min_val_epoch, min_val_loss),
             xytext=(min_val_epoch+1, min_val_loss+0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

plt.show()

# %%
# Prediction on test set
model.eval()
nn_predicted = []
test_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted_batch = torch.max(outputs, 1)
        _, labels_batch = torch.max(labels, 1)
        
        nn_predicted.extend(predicted_batch.cpu().numpy())
        test_targets.extend(labels_batch.numpy())

# %%
# Model evaluation
print("Neural Network Results:")
print(classification_report(test_targets, nn_predicted))

# %%
# Enhanced confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_targets, nn_predicted)
labels = ['Real (0)', 'Fake (1)']

# Calculate percentages for display
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Define custom annotations
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', linewidths=1, linecolor='black',
                 xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 12})

plt.title('Neural Network Confusion Matrix', fontsize=16, pad=20)
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Actual Values', fontsize=14)
plt.tight_layout()

# Add global accuracy
accuracy = np.trace(cm) / np.sum(cm) * 100
plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy:.2f}%', ha='center', fontsize=12)

plt.show()

# %%
# ROC curve visualization
model.eval()
nn_y_true = []
nn_y_pred_proba = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        _, labels_batch = torch.max(labels, 1)
        nn_y_true.extend(labels_batch.numpy())
        nn_y_pred_proba.extend(outputs.cpu().numpy()[:, 1])

# Calculate the ROC curve
nn_fpr, nn_tpr, _ = roc_curve(nn_y_true, nn_y_pred_proba)
nn_roc_auc = metrics.auc(nn_fpr, nn_tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(nn_fpr, nn_tpr, color='darkorange', lw=2, label=f'Neural Network (AUC = {nn_roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Reference Line')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Neural Network ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# %%
# Classification metrics visualization by class
nn_report = classification_report(test_targets, nn_predicted, output_dict=True)
nn_report_df = pd.DataFrame(nn_report).transpose()
nn_report_df = nn_report_df.drop('accuracy', errors='ignore')

plt.figure(figsize=(12, 8))
ax = sns.heatmap(nn_report_df.iloc[:-1, :].astype(float), annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Neural Network Classification Metrics by Class', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# # XGBoost Model

# %%
# Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Create a validation set for XGBoost
from sklearn.model_selection import train_test_split
X_train_raw = scaler_x.inverse_transform(X_train)  # Get unscaled features back
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train_raw, y_train, test_size=0.1, random_state=42
)

# Train the model with early stopping
eval_set = [(X_val_xgb, y_val_xgb)]
xgb_model.fit(
    X_train_xgb, 
    y_train_xgb,
    eval_set=eval_set,
    verbose=True,
)

# %% [markdown]
# # XGBoost Model Validation and Results

# %%
# Get XGBoost predictions
X_test_raw = scaler_x.inverse_transform(X_test)  # Get unscaled features back
xgb_predicted = xgb_model.predict(X_test_raw)
xgb_predicted_proba = xgb_model.predict_proba(X_test_raw)[:, 1]

# %%
# Model evaluation
print("XGBoost Results:")
print(classification_report(y_test, xgb_predicted))

# %%
# XGBoost confusion matrix
plt.figure(figsize=(10, 8))
cm_xgb = confusion_matrix(y_test, xgb_predicted)
labels = ['Real (0)', 'Fake (1)']

# Calculate percentages for display
cm_percent_xgb = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis] * 100

# Define custom annotations
annot_xgb = np.empty_like(cm_xgb).astype(str)
for i in range(cm_xgb.shape[0]):
    for j in range(cm_xgb.shape[1]):
        annot_xgb[i, j] = f'{cm_xgb[i, j]}\n({cm_percent_xgb[i, j]:.1f}%)'

ax = sns.heatmap(cm_xgb, annot=annot_xgb, fmt='', cmap='Greens', linewidths=1, linecolor='black',
                 xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 12})

plt.title('XGBoost Confusion Matrix', fontsize=16, pad=20)
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Actual Values', fontsize=14)
plt.tight_layout()

# Add global accuracy
accuracy_xgb = np.trace(cm_xgb) / np.sum(cm_xgb) * 100
plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy_xgb:.2f}%', ha='center', fontsize=12)

plt.show()

# %%
# XGBoost ROC curve
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_predicted_proba)
xgb_roc_auc = metrics.auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(10, 8))
plt.plot(xgb_fpr, xgb_tpr, color='forestgreen', lw=2, label=f'XGBoost ROC (AUC = {xgb_roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Reference Line')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('XGBoost ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# %%
# XGBoost classification metrics by class
xgb_report = classification_report(y_test, xgb_predicted, output_dict=True)
xgb_report_df = pd.DataFrame(xgb_report).transpose()
xgb_report_df = xgb_report_df.drop('accuracy', errors='ignore')

plt.figure(figsize=(12, 8))
ax = sns.heatmap(xgb_report_df.iloc[:-1, :].astype(float), annot=True, cmap='BuGn', fmt='.3f')
plt.title('XGBoost Classification Metrics by Class', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Feature importance visualization for XGBoost
plt.figure(figsize=(12, 8))
feature_names = instagram_df_train.drop(columns=['fake']).columns
xgb_importance = xgb_model.feature_importances_
indices = np.argsort(xgb_importance)[::-1]

plt.title('XGBoost Feature Importance', fontsize=16)
plt.bar(range(len(xgb_importance)), xgb_importance[indices], color='forestgreen', align='center')
plt.xticks(range(len(xgb_importance)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Model Comparison

# %%
# Compare ROC curves of both models
plt.figure(figsize=(12, 8))
plt.plot(nn_fpr, nn_tpr, 'darkorange', lw=2, label=f'Neural Network (AUC = {nn_roc_auc:.3f})')
plt.plot(xgb_fpr, xgb_tpr, 'forestgreen', lw=2, label=f'XGBoost (AUC = {xgb_roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Reference Line')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve Comparison: Neural Network vs XGBoost', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %%
# Compare accuracy, precision, recall, and F1-score for both models
metrics_comparison = pd.DataFrame({
    'Model': ['Neural Network', 'XGBoost'],
    'Accuracy': [accuracy_score(test_targets, nn_predicted), accuracy_score(y_test, xgb_predicted)],
    'Precision (Class 1)': [nn_report['1']['precision'], xgb_report['1']['precision']],
    'Recall (Class 1)': [nn_report['1']['recall'], xgb_report['1']['recall']],
    'F1-Score (Class 1)': [nn_report['1']['f1-score'], xgb_report['1']['f1-score']],
})

metrics_comparison.set_index('Model', inplace=True)

plt.figure(figsize=(14, 8))
ax = sns.heatmap(metrics_comparison, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.title('Model Performance Comparison: Neural Network vs XGBoost', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Bar chart comparison of key metrics
metrics_df_melted = pd.melt(metrics_comparison.reset_index(), id_vars=['Model'])

plt.figure(figsize=(14, 8))
ax = sns.barplot(x='variable', y='value', hue='Model', data=metrics_df_melted)
plt.title('Model Performance Comparison: Neural Network vs XGBoost', fontsize=16)
plt.xlabel('Metric', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend(title='Model')
plt.ylim(0, 1.0)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=10)

plt.tight_layout()
plt.show()


