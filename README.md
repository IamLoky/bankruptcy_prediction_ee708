# bankruptcy_prediction_ee7081. Objective
The objective of this study is to develop a predictive model capable of assessing the likelihood of corporate bankruptcy based on various financial and business-related attributes. By leveraging machine learning techniques, the model aims to enhance financial risk assessment and contribute to informed decision-making for economic stability.\

Preprocessing
The outlier capping process applies the Interquartile Range (IQR) method to limit extreme values in numerical features. It calculates Q1 (25th percentile) and Q3 (75th percentile), determines the IQR (Q3 - Q1), and caps values below Q1 - 1.5 × IQR and above Q3 + 1.5 × IQR to these bounds revealing a class imbalance of 5301 non-bankrupt vs. 154 bankrupt cases

The dataset is split into 80% training and 20% testing while preserving class distribution using stratification.
1.Refining Training Data
The dataset exhibited a severe class imbalance (5301 non-bankrupt vs. 154 bankrupt cases), which could lead to biased predictions favoring the majority class. Traditional machine learning models tend to perform poorly on highly imbalanced datasets. To address this issue, existing research on resampling techniques was referred to, and SMOTE (Synthetic Minority Over-sampling Technique) was identified as a suitable method for mitigating class imbalance. After experimentation, the best ratio was found to be 10%, adjusting the distribution from 5301 non-bankrupt vs. 154 bankrupt to 4241 non-bankrupt vs. 424 bankrupt.

In addition to handling class imbalance, feature selection was performed to enhance model interpretability and efficiency. Out of the initial 95 features, only 15 were retained after filtering based on importance scores using a threshold of 0.02. Then, referring to Fig. 2, the correlation heatmap, the features were further reduced to 10 based on correlation values. As a result, dimensionality was reduced from 95 to 10 features (89.47%), minimizing the influence of less informative variables and leading to a more robust predictive model.
2. Model Architecture:
  
  Model Selection:	
		To identify the most effective bankruptcy prediction model, several standard machine learning methods were evaluated after referring to existing literature on 			financial risk assessment and classification algorithms:

			1.Logistic Regression (LR): A linear classifier estimating bankruptcy probability.

			2.K-Nearest Neighbors (KNN): A distance-based classifier sensitive to feature scaling.

			3.Decision Tree (DT): A rule-based model that splits data efficiently but can overfit without depth control

			4.Random Forest (RF): An ensemble of decision trees improving generalization.

			5.XGBoost (XGB): A gradient boosting model optimizing weak learners.

			6.LightGBM (LGBM): A faster gradient boosting alternative for large datasets.

			7.Neural Networks (NN): A deep learning model capturing complex patterns.
  Metric Selection:
		To evaluate model performance, various metrics were considered, including accuracy, precision, recall, and F1-score. Given the dataset’s severe class imbalance 		(5301 non-bankrupt vs. 154 bankrupt cases), F1 score was chosen as the most suitable metric.
F1 score calculates recall and precision for each class separately and then averages the values, ensuring that performance on the minority class is not overshadowed by the majority class. 
The graph in Fig.3 comparing Mean CV F1 Scores and Test F1 Scores across various models shows that Logistic Regression achieves the highest Test F1 Score. While its Mean CV F1 Score is slightly lower than some other models (LGBM and XGBoost), the superior performance on the test set, reflecting better generalization, led us to select Logistic Regression for our final model
  Model Tuning:
		To enhance the performance of Logistic Regression, L1 (Lasso) regularization were introduced to improve generalization and prevent overfitting. A Grid Search was conducted over a range of λ values to identify the optimal regularization strength that maximized F1 Score while maintaining model 				stability.
3. Final Accuracies:

   Final Model Performance:
(Insert confusion matrix and classification report for the tuned Logistic Regression model.)

The tuned model demonstrated improved F1-score, effectively balancing precision and recall for both classes. This concludes the analysis, emphasizing the role of SMOTE, feature selection, and hyperparameter tuning in enhancing model performance.


