import pandas as pd
import joblib

model= joblib.load('./models/final_model.pkl')

# new_data = pd.DataFrame([{
#     'age': 58,
#     'sex': 1,
#     'cp': 2,
#     'thalach': 150,
#     'exang': 0,
#     'oldpeak': 1.2,
#     'ca': 0,
#     'thal': 3
# }])

new_data = pd.DataFrame([{
    'age': 45,
    'sex': 1,           # 1 = male
    'cp': 0,            # typical angina
    'thalach': 165,     # good heart rate
    'exang': 0,         # no exercise-induced angina
    'oldpeak': 0.0,     # no ST depression
    'ca': 0,            # no major vessels colored
    'thal': 2           # normal thalassemia
}])

prediction= model.predict(new_data)
probability = model.predict_proba(new_data)

print(" Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
