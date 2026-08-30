from pathlib import Path
from importlib import import_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import HTTPException
import joblib
import pandas as pd


column_transformer = import_module("sklearn.compose._column_transformer")

if not hasattr(column_transformer, "_RemainderColsList"):
    setattr(column_transformer, "_RemainderColsList", type(
        "_RemainderColsList", (list,), {}
    ))

model_path = Path(__file__).resolve().parent / "logistic_regression_model.joblib"
model = joblib.load(model_path)

app = FastAPI(title="Student At Risk System", version="1.0.0")

# Allow the front-end to call this API. Using permissive CORS for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI files (index.html) from the project root so the page and API share origin.
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent, html=True), name="static")


@app.get('/')
def root_index():
    index_file = Path(__file__).resolve().parent / "index.html"
    return FileResponse(index_file)

@app.get('/health')
def health():
    return {"message": "Student At Risk System API is healthy"}


@app.post("/predict")
def predict_student_at_risk(student_data: dict):
    # Validate input locally to avoid importing helpers (prevents circular imports)
    from pydantic import BaseModel

    class StudentInputDataLocal(BaseModel):
        studytime: int
        failures: int
        schoolsup: str
        famsup: str
        activities: str
        higher: str
        internet: str
        famrel: int
        health: int
        absences: int

    def build_result(prediction: int, probability: float):
        result = "At risk" if prediction == 1 else "Not at risk"
        prob_at_risk = round(probability[1], 3)
        prob_not_at_risk = round(probability[0], 3)
        prob_percentage = round(prob_at_risk * 100, 2) if result == "At risk" else round(prob_not_at_risk * 100, 2)
        return {
            "result": result,
            "probability": {"at_risk": prob_at_risk, "not_at_risk": prob_not_at_risk},
            "model_percentage": prob_percentage,
            "human_required": True,
        }

    try:
        validated = StudentInputDataLocal(**student_data)
        data_frame = pd.DataFrame([validated.model_dump()])
        prediction = model.predict(data_frame)[0]
        prob = model.predict_proba(data_frame)[0]
        return build_result(prediction, prob)
    except Exception as e:
        import traceback, sys
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))