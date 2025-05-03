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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split

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
# Load the dataset
df = pd.read_csv('kaggle/input/Dataset_2/LIMFADD.csv')

# %% [markdown]
# # Statistical Analysis

# %%
df.head()

# %%
df.tail()

# %%
# Getting dataframe info
df.info()

# %%
# Get the statistical summary of the dataframe
df.describe()

# %%
# Check for missing values
df.isnull().sum()

# %%
# Check for duplicates
df.duplicated().sum()

# %%
# Get the number of unique values in the "Labels" column (Target column)
df['Labels'].value_counts()

# %% [markdown]
# # Data Visualization

# %%
# Visualization of labels distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Labels', data=df, palette=['#3498db', '#e74c3c'])
plt.title('Instagram Profile Distribution: Labels', fontsize=16)
plt.xlabel('Profile Labels', fontsize=12)
plt.ylabel('Number of Profiles', fontsize=12)
# Add values on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom', fontsize=12)

plt.show()

# %%
# Enhanced visualization of the followers distribution
plt.figure(figsize=(14, 8))
sns.histplot(df['Followers'], bins=30, kde=True, color='#3498db')
plt.title('Followers Distribution', fontsize=16)
plt.xlabel('Number of Followers', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add a vertical line for the mean
mean_val = df['Followers'].mean()
plt.axvline(x=mean_val, color='#e74c3c', linestyle='--', linewidth=2)
plt.text(mean_val + 0.02, plt.ylim()[1]*0.9, f'Mean: {mean_val:.2f}', color='#e74c3c', fontsize=12)

plt.show()

# %%
# Following vs Followers with Labels
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Following', y='Followers', hue='Labels', data=df, palette='viridis', alpha=0.8, s=100)
plt.title('Following vs Followers by Profile Type', fontsize=16)
plt.xlabel('Following Count', fontsize=14)
plt.ylabel('Followers Count', fontsize=14)
plt.legend(title='Labels')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Profile Picture vs Bio with Labels
plt.figure(figsize=(12, 8))
sns.countplot(x='Profile Picture', hue='Labels', data=df, palette='Set2')
plt.title('Profile Picture Presence by Label Type', fontsize=16)
plt.xlabel('Has Profile Picture', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Labels')
plt.tight_layout()
plt.show()

# %%
# Enhanced visualization of the correlation matrix
plt.figure(figsize=(16, 14))
mask = np.triu(df.corr(numeric_only=True))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap=cmap, linewidths=0.5, 
            mask=mask, vmin=-1, vmax=1, center=0, square=True, cbar_kws={"shrink": .8})

plt.title('Feature Correlation Matrix', fontsize=18, pad=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Data Preprocessing

# %%
# Drop the derived features as they contain redundant information
df.drop(['Following/Followers', 'Posts/Followers'], axis=1, inplace=True)

# %%
# Label Encoding for categorical features
le = LabelEncoder()
categorical_cols = ['Bio', 'Profile Picture', 'External Link', 'Threads', 'Labels']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# %%
# Prepare data for modeling
X = df.drop('Labels', axis=1)
y = df['Labels']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"Y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"Y_test Shape: {y_test.shape}")

# %%
# Scale the data before training the models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # Neural Network Model

# %%
# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# Convert labels to one-hot encoding
y_train_tensor = torch.zeros(len(y_train), len(np.unique(y_train)))
y_train_tensor[range(len(y_train)), y_train.astype(int)] = 1
y_test_tensor = torch.zeros(len(y_test), len(np.unique(y_test)))
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
    def __init__(self, input_dim=X_train_scaled.shape[1], num_classes=len(np.unique(y_train))):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.softmax(self.layer4(x))
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
# # Neural Network Model Validation and Results

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
num_classes = len(np.unique(test_targets))
labels = [f'Class {i}' for i in range(num_classes)]

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
# ROC curve visualization for multiclass using one-vs-rest approach
model.eval()
nn_y_true = []
nn_y_pred_proba = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        _, labels_batch = torch.max(labels, 1)
        nn_y_true.extend(labels_batch.numpy())
        nn_y_pred_proba.extend(outputs.cpu().numpy())

nn_y_pred_proba = np.array(nn_y_pred_proba)
nn_y_true = np.array(nn_y_true)

# Plot ROC curves for each class (one-vs-rest)
plt.figure(figsize=(10, 8))
num_classes = len(np.unique(nn_y_true))
colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

for i, color in zip(range(num_classes), colors):
    # For each class, calculate ROC curve (one-vs-rest)
    nn_fpr, nn_tpr, _ = roc_curve((nn_y_true == i).astype(int), nn_y_pred_proba[:, i])
    nn_roc_auc = metrics.auc(nn_fpr, nn_tpr)
    plt.plot(nn_fpr, nn_tpr, color=color, lw=2,
             label=f'Class {i} (AUC = {nn_roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Neural Network ROC Curves (One-vs-Rest)', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Classical Machine Learning Models

# %%
# Import machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# %%
# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# %%
# Train and evaluate each model
results = []
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    
    # Print results
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {recall:.4f}")
    
    # Save results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Convert results to dataframe
results_df = pd.DataFrame(results)

# %%
# Display summary table
print("\nModel Performance Summary:")
results_df.set_index('Model', inplace=True)
results_df

# %%
# Enhanced visualization of model comparison
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Accuracy', data=results_df.reset_index(), palette='viridis')
plt.title('Machine Learning Models Accuracy Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, v in enumerate(results_df['Accuracy']):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Compare all metrics across models
metrics_data = results_df.reset_index().melt(id_vars=['Model'], 
                                          value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                          var_name='Metric', value_name='Value')

# Adjust Accuracy to be on same scale as other metrics
metrics_data.loc[metrics_data['Metric'] == 'Accuracy', 'Value'] = metrics_data.loc[metrics_data['Metric'] == 'Accuracy', 'Value'] / 100

plt.figure(figsize=(16, 10))
sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_data, palette='Set2')
plt.title('Model Performance Metrics Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.ylim(0, 1.0)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Model Comparison: Neural Network vs ML Models

# %%
# Get neural network metrics
nn_accuracy = accuracy_score(test_targets, nn_predicted) * 100
nn_precision = metrics.precision_score(test_targets, nn_predicted, average='macro')
nn_recall = metrics.recall_score(test_targets, nn_predicted, average='macro')
nn_f1 = metrics.f1_score(test_targets, nn_predicted, average='macro')

# Add Neural Network to results
all_models = results_df.reset_index().copy()
nn_row = pd.DataFrame([{
    'Model': 'Neural Network',
    'Accuracy': nn_accuracy,
    'Precision': nn_precision,
    'Recall': nn_recall,
    'F1 Score': nn_f1
}])
all_models = pd.concat([all_models, nn_row], ignore_index=True)

# %%
# Create final comparison visualization
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Accuracy', data=all_models, palette='coolwarm')
plt.title('All Models Accuracy Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, v in enumerate(all_models['Accuracy']):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Get the best model
best_model_idx = all_models['Accuracy'].idxmax()
best_model_name = all_models.loc[best_model_idx, 'Model']
best_model_accuracy = all_models.loc[best_model_idx, 'Accuracy']

print(f"Best Model: {best_model_name} with accuracy {best_model_accuracy:.2f}%")


