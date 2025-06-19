import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "../models/sentiment_model.pkl")

model = joblib.load(model_path)

def get_prediction(text: str):
    return model.predict([text])[0]

if __name__ == "__main__":
    user_str = input("Enter any string: ")
    result = get_prediction(user_str)
    result = "Positive" if result == 1 else "Negative"
    print(result)