# evaluate.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
results={} 
def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
    print("Classification Report:\n", classification_report(y_test, y_predict))
    print("-" * 40)

    return model.__class__.__name__, acc