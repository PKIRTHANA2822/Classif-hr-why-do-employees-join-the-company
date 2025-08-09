# Project title
- Predicting whether a candidate will join — Why do employees join the company? (Classification: Joined vs Not Joined)
<img width="760" height="422" alt="employee image" src="https://github.com/user-attachments/assets/ee33e6ec-b424-43cd-95b0-317f6eb78b95" />
# Objective
- Build a predictive model that estimates the probability a candidate who accepted an offer will actually join the company. Provide interpretable insights for HR to reduce no-shows and prioritise interventions.
# Why we do this project
- Reduce cost & time lost to re-opened hiring processes.
- Prioritise offers and follow-ups for high-risk candidates.
- Provide HR with actionable features (e.g., notice period, offered hike, joining bonus) that influence joining probability.
# Step-by-step approach (high level)
- Data ingestion & basic cleaning
- Exploratory Data Analysis (EDA) and business insights
- Correct preprocessing pipeline (no data leakage)
- Feature engineering & selection
- Model training (baseline → tuned ensembles) using proper CV + resampling in pipeline
- Model evaluation (ROC-AUC, PR-AUC, calibration, confusion matrix)
- Interpretability (SHAP / feature importances)
- Deliverables & recommendations for HR
- Exploratory Data Analysis — what to cover
- Class balance: Status.value_counts() (plot bar/pie).
- Numerical distributions vs Status: histograms / KDEs for Duration to accept offer, Age, Rex in Yrs, Percent difference CTC.
- Categorical counts vs Status: countplot for Notice period, Offered band, Joining Bonus, LOB, Location, Candidate Source.
- Pivot tables for business insights (e.g., joining% by Notice period, Offered band).
- Visual: stacked bar charts and annotated pivot tables for HR presentation.
- (You already did most of this — good.)
- Preprocessing — correct, leakage-free flow (recommended)
# Key rules:
- Fit encoders / scalers only on training data. Use .transform() on test.
- Do not resample the test set. Resampling (SMOTE, SMOTEENN) should be applied only inside the training pipeline.
- Use ColumnTransformer + imblearn.Pipeline to keep preprocessing, resampling and model together (avoids leakage during CV).
# Recommended code (cleaned and minimal):
- python
- Copy
- Edit
- from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
- from sklearn.compose import ColumnTransformer
- from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
- from imblearn.pipeline import Pipeline as ImbPipeline
- from imblearn.combine import SMOTEENN
- from sklearn.ensemble import RandomForestClassifier
# split once
- X = df.drop(['Candidate Ref','Status'], axis=1)
- y = df['Status'].map({'Joined':1, 'Not Joined':0})
- X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
- num_cols = [c for c in X_trn.columns if X_trn[c].dtype in ['int64','float64']]
- cat_cols = [c for c in X_trn.columns if X_trn[c].dtype == 'object' and c != 'Status']
- preproc = ColumnTransformer([
    ('num', MinMaxScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False), cat_cols)
])

pipeline = ImbPipeline([
    ('preproc', preproc),
    ('resample', SMOTEENN(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

# example param grid (tune the clf__... params)
- param_grid = {'clf__n_estimators': [100,150], 'clf__max_depth':[8,10]}
- cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- gs = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
- gs.fit(X_trn, y_trn)
# evaluate on untouched test set
- pred = gs.predict(X_tst)
- pred_proba = gs.predict_proba(X_tst)[:,1]
- Feature engineering ideas (specific to your dataset)
- Percent difference CTC is already informative — keep as numeric and consider non-linear transforms (log, bins).
- Duration to accept offer → bin into fast/normal/slow acceptors (e.g., 0–7, 8–21, 22+ days).
- Notice period → treat as ordinal or bucket (0, 30, 60, 90+) or convert to notice_bucket.
- Offered band → ordinal encoding (E0 < E1 < E2...) if bands imply ordering.
- Joining Bonus → binary flag (0/1).
- Location high-cardinality → use target encoding or frequency encoding (or group by city tier).
- Interaction features: percent_gap = Percent hike expected - Percent hike offered (if not already), age_x_experience = Age * Rex in Yrs
# Date features: parse DOJ Extended into Boolean or days-from-offer if meaningful.
- Feature selection (methods)
- Filter methods: correlation matrix for continuous features; chi2 or mutual_info_classif for categorical vs target.
- Wrapper methods: RF.feature_importances_, RFE with a simple estimator.
- Embedded: L1-penalized Logistic Regression for sparsity.
- Always validate selected features with cross-validation — i.e., do selection inside CV pipeline to avoid optimistic bias.
# Code example (feature importance quick check):
- python
- Copy
- Edit
- rf = RandomForestClassifier(random_state=42)
- rf.fit(X_train_transformed, y_train_resampled)
- pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
- Modeling — recommended models & tuning
- Baseline → Logistic Regression (calibrated probabilities)
- Tree ensembles → RandomForest, GradientBoosting (or XGBoost/LightGBM)
# Tuning strategy:
- Use imblearn.Pipeline with preproc, resample, and classifier.
- Use RandomizedSearchCV for a broad sweep, then GridSearchCV on promising ranges.
- CV: StratifiedKFold(n_splits=5) and scoring: roc_auc (primary), also track average_precision (PR-AUC) for imbalanced tasks.
# Hyperparameter examples (names to use inside Grid for pipeline):
'clf__n_estimators', 'clf__max_depth', 'clf__max_features', 'clf__class_weight'
Evaluation & testing (what to report)
- ROC-AUC, PR-AUC (average precision), accuracy, precision, recall, F1.
- Confusion matrix with business-oriented thresholds: compute at default 0.5 and at risk-based thresholds.
- Calibration curve and Brier score if you will use predicted probabilities to prioritise outreach.
- Fairness checks: performance by Location, Gender, LOB to detect bias.
### Stability: cross-validated performance and variance (std dev).
# Threshold tuning (business metric example):
- python
- Copy
- Edit
- from sklearn.metrics import precision_recall_curve
- prec, rec, th = precision_recall_curve(y_true, y_scores)
# choose threshold balancing recall (catching those likely to not join) vs precision (effort)
- Interpretability (SHAP + feature importances)
- Tips & correct usage (fixes from your earlier errors):
- If you use ColumnTransformer + OneHotEncoder, get feature_names with preproc.get_feature_names_out() and convert to str.
- shap.TreeExplainer on a fitted tree model will return shap_values. For binary classification explain.shap_values(X) often returns a list [arr_for_class0, arr_for_class1]. Use class 1 array for positive class (shap_values[1]).
- Ensure features passed to shap.summary_plot is a DataFrame with string column names (no ints), otherwise len() errors appear.
# Common SHAP gotchas you had in your notebook (and fixes):
- Don’t fit_transform encoder on test data — use transform.
- Convert X columns to strings for SHAP labels.
- If shap.summary_plot complains object of type 'int' has no len(), that usually means one or more column names are integers — fix by ensuring X.columns = X.columns.astype(str).
- Pitfalls / gotchas (I noticed in your notebook)
- You used OH_encoder.fit_transform on the test set; that leaks information. Use transform on test only.
- You resampled the test set with SMOTEENN — do not resample test data. Test set must reflect real-world distribution.
- When scaling, fit only on train, then transform test.
- When using RandomizedSearchCV or GridSearchCV, ensure resampling is inside the pipeline so CV folds are resampled independently (no leakage).
# Output / Deliverables (what to hand to stakeholders)
<img width="1398" height="528" alt="Screenshot 2025-08-09 211257" src="https://github.com/user-attachments/assets/18df1f22-19b1-4ecd-b5b9-9364c969e8cb" />
