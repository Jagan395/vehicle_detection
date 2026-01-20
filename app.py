from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from backend.Model_Detection import detection

app = FastAPI()

@app.get("/")
def home():
    return {"message":"vehicle_detection"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    result = detection(file)
    return JSONResponse(content=result)
