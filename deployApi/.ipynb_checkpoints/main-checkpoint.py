from fastapi import FastAPI, HTTPException
import pickle5 as pickle
import schemas

app = FastAPI(title="Iris Prediction API",
    description="Data Pipeline using AI platform pipeline + Cloud Run + fastapi",
    version="1.0.1",
)


with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

classes = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']

@app.post("/predict'")
def predict(request:schemas.Iris):
    try:
        predictmodel = model.predict([[
            request.sepal_length, 
            request.sepal_width, 
            request.petal_length, 
            request.petal_width
        ]])
        return {"Predict": classes[predictmodel[0]]}
    except:
        raise HTTPException(status_code=404)
