# Precision-Medicine-Genomics-Data-Science
A "Personalized Precision Medicine &amp; Geonomics" inform potential treatment outcomes for new patients, aiding in personalized medicine or clinical decision-making.

How it Informs Personalized Medicine and Clinical Decision-Making:
1. Personalized Risk/Response Assessment: By inputting a new patient's gene expression data (and other relevant clinical features), the model can predict their likely TreatmentResponse (e.g., High risk, Moderate risk, Low risk). This allows clinicians to tailor treatment plans to individual patients rather than applying a one-size-fits-all approach.
2. Early Intervention/Modification: If the model predicts a low likelihood of response to a standard treatment, clinicians could consider alternative therapies earlier, potentially saving time, reducing patient suffering, and avoiding ineffective treatments.

Below are the summary of key steps along with snapshots of the input dataset, output results and charts of trained model.

1. Input Training Dataset Loading - 
   * We loaded the 'Gene Expression Analysis and Disease Relationship.csv' dataset sourced from Kaggle (https://www.kaggle.com/datasets/ylmzasel/gene-expression-analysis-and-disease-relationship/data).
   * Gene expression features (X) were separated from the TreatmentResponse target variable (y).
   * The features were StandardScaler-normalized to ensure proper scaling for PCA.

Input Dataset - 
<img width="746" height="306" alt="Screenshot 2026-04-12 at 2 23 34 PM" src="https://github.com/user-attachments/assets/86d0d390-b401-4014-b979-16daf55099f3" />

2. Dimensionality Reduction with PCA:
   * Principal Component Analysis (PCA) was applied to reduce the dimensionality of the gene expression data.
   * We retained components that explained 95% of the variance, resulting in 4 principal components (PC_0 to PC_3) from the original 7 features.
   * The 'Scree Plot' visually confirmed the cumulative explained variance, showing how these 4 components capture most of the data's variability.
<img width="690" height="387" alt="Screenshot 2026-04-12 at 2 24 54 PM" src="https://github.com/user-attachments/assets/9ac67f96-24fa-4f50-a3d8-be77a25c8682" />

3. Gradient Boosting Model Training (XGBoostClassifier):
   * We proceeded with an XGBoostClassifier to predict TreatmentResponse based on the 4 PCA components.
   * The data was split into training and testing sets (80/20).
   * The model achieved a high accuracy of 0.99 on the test set.
   * The classification report revealed strong precision, recall, and F1-scores, particularly for the majority class (Class 2). We noted the class imbalance in the target variable, where Class 2 dominated.

Accuracy result of the model post training with 80% of dataset and testing with 20% of dataset 
<img width="1265" height="375" alt="Screenshot 2026-04-12 at 2 26 58 PM" src="https://github.com/user-attachments/assets/7f608a32-beda-4674-98c3-1dc5001f7e5a" />

4. SHAP (SHapley Additive exPlanations) Bar and Dot Plot to visualize which pca component is influencing components, consistently driving the model's predictions.
   * SHAP Bar Plot: This showed the mean absolute SHAP value for each PC, indicating its average impact on the model's output. PC_0 consistently appeared as the most important feature, followed by PC_1 and PC_3. PC_2 showed very little average impact.
   * SHAP Dot Plot: This visualization displayed the individual SHAP values for each prediction. It confirmed that while PC_0 and PC_1 had a wide range of strong individual impacts, PC_2's individual impacts were consistently very small, clustering around zero. PC_3 showed more individual effects than PC_2, even if not as pronounced as PC_0 and PC_1.
   <img width="962" height="968" alt="Screenshot 2026-04-12 at 2 35 26 PM" src="https://github.com/user-attachments/assets/68c21fba-f80e-4139-9075-5054fac21e48" />


To test the model in local machine, follow below steps - 


1. Clone the repository and ensure your directory structure matches the provided repo structure.


2. Environment Setup
Create a virtual environment to manage dependencies and avoid system conflicts:

# Create the virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt


3. Train the Model (Generate Artifacts)
Before running the API, you must train the model locally to generate the .pkl files (Scaler, PCA, and XGBoost).

python3 notebooks/train_model.py
Wait for the message: Training Complete. Models saved to /models directory.


4. Launch the FastAPI Server
Now, start the Uvicorn server to host the local API:

uvicorn app.main:app --reload
The API will be live at: http://127.0.0.1:8000


5. Testing the API
You can test the model using the interactive Swagger UI:

Open http://127.0.0.1:8000/docs in your browser.

Navigate to the POST /predict endpoint.

Click "Try it out" and input a patient profile (7 features), for example:

JSON
{
  "gene_expression_values": [1002, 75.0, 1.0, 8.5, 7.1, 2.0, 2.0]
}
Click Execute to view the risk stratification and clinical guidance.
