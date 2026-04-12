import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
# Load the dataset you downloaded
# Replace the filename if it differs slightly
file_path = '/Gene Expression Analysis and Disease Relationship.csv'
df = pd.read_csv(file_path)

# Separate features (genes) and the target variable (disease/outcome)
# Assuming 'target' or similar is the last column; adjust based on .head()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"Original Dataset Shape: {X.shape}") # Shows (patients, genes)
print(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled)
pca = PCA(n_components=0.95)
#print(pca)
X_pca = pca.fit_transform(X_scaled)
#print(X_pca)
print(f"Reduced Dataset Shape: {X_pca.shape}")
print(f"Number of components needed to explain 95% variance: {pca.n_components_}")

# 3. Visualize the "Scree Plot" (Great for your GitHub README)
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Variance Coverage')
plt.grid(True)

# Set x-ticks to match the number of components (1-indexed)
plt.xticks(ticks=range(len(pca.explained_variance_ratio_)), labels=range(1, len(pca.explained_variance_ratio_) + 1))

plt.show()

# Get the feature names from the original DataFrame X
original_feature_names = X.columns

# Create a DataFrame to display the PCA components (loadings)
pca_components_df = pd.DataFrame(pca.components_, columns=original_feature_names,
                                 index=[f'PC_{i}' for i in range(pca.n_components_)])

print("\nContribution of Original Features to Each Principal Component:")
display(pca_components_df)

print("\nInterpretation:")
print("Each row represents a Principal Component (PC) and each column represents an original feature.")
print("The values (loadings) indicate the strength and direction of the relationship between the original feature and the PC.")
print("Higher absolute values indicate a stronger contribution. Positive values mean they move in the same direction, negative values in opposite directions.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder = False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nUnique values in target variable 'y' (TreatmentResponse): {y.unique()}")
print(f"Value counts for target variable 'y':\n{y.value_counts()}")



# 1. Initialize the SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Get feature names for clarity
feature_names = [f"PC_{i}" for i in range(X_pca.shape[1])]

# 2. Visualize Feature Importance with a larger figure size (Bar Plot - Mean Absolute SHAP)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
plt.title('SHAP Bar Plot of Feature Importance (Mean Absolute SHAP)')
plt.tight_layout()
plt.show()

# 3. Visualize Feature Importance with a larger figure size (Default Dot Plot - Individual SHAP values)
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Dot Plot of Feature Importance (Individual SHAP Values)')
plt.tight_layout()
plt.show()
