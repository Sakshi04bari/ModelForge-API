import joblib
from sklearn.datasets import load_iris

model = joblib.load("ml/saved_model/model.joblib")

iris = load_iris()

#sample = [[5.1, 3.5, 1.4, 0.2]]
#sample = [[6.0, 2.9, 4.5, 1.5]]
sample = [[6.5, 3.0, 5.5, 1.8]]
prediction = model.predict(sample)[0]

predicted_name = iris.target_names[prediction]

print("Predicted class:", prediction)
print("Predicted flower:", predicted_name)