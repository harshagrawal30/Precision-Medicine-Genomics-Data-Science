# Precision-Medicine-Genomics-Data-Science-
A "Personalized Precision Medicine &amp; Geonomics" inform potential treatment outcomes for new patients, aiding in personalized medicine or clinical decision-making.

How it Informs Personalized Medicine and Clinical Decision-Making:
1. Personalized Risk/Response Assessment: By inputting a new patient's gene expression data (and other relevant clinical features), the model can predict their likely TreatmentResponse (e.g., complete response, partial response, no response). This allows clinicians to tailor treatment plans to individual patients rather than applying a one-size-fits-all approach.
2. Early Intervention/Modification: If the model predicts a low likelihood of response to a standard treatment, clinicians could consider alternative therapies earlier, potentially saving time, reducing patient suffering, and avoiding ineffective treatments.

Below are the snapshots for the input dataset and the output results and charts.

1. Input Training Dataset Preview
<img width="746" height="306" alt="Screenshot 2026-04-12 at 2 23 34 PM" src="https://github.com/user-attachments/assets/86d0d390-b401-4014-b979-16daf55099f3" />

2. PCA 4-Component Variance Coverage (covering 95% of data variance)
<img width="690" height="387" alt="Screenshot 2026-04-12 at 2 24 54 PM" src="https://github.com/user-attachments/assets/9ac67f96-24fa-4f50-a3d8-be77a25c8682" />

3. Accracy result of the model post training with 80% of dataset and testing with 20% of dataset 
<img width="1265" height="375" alt="Screenshot 2026-04-12 at 2 26 58 PM" src="https://github.com/user-attachments/assets/7f608a32-beda-4674-98c3-1dc5001f7e5a" />

4. SHAP Bar and Dot Plot to visualize which pca component is influencing components, consistently driving the model's predictions.

   <img width="962" height="968" alt="Screenshot 2026-04-12 at 2 35 26 PM" src="https://github.com/user-attachments/assets/68c21fba-f80e-4139-9075-5054fac21e48" />
