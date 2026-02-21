import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
# Ensure cancer_data.csv is in the same folder!
df = pd.read_csv("cancer_data.csv")

# 2. Select the 8 features and the target
X = df[['age', 'gender', 'bmi', 'smoking', 'genetic_risk', 
        'physical_activity', 'alcohol_intake', 'cancer_history']]
y = df["diagnosis"]

# 3. Split the data (25% test size as per your notebook)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# 4. Initialize the "Best" Random Forest Model (Task 4 Params)
rf_model_custom = RandomForestClassifier(
    n_estimators=50,
    max_features="log2",
    criterion="entropy",
    bootstrap=False,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=5 # Added for consistency
)

# 5. Train the model
rf_model_custom.fit(X_train, y_train)

# 6. SAVE THE TWO PKL FILES
# Save the actual model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf_model_custom, model_file)

# Save the feature names to ensure app.py uses the right order
with open('features.pkl', 'wb') as features_file:
    pickle.dump(list(X.columns), features_file)

print("Success! 'model.pkl' and 'features.pkl' have been generated.")