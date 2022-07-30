from typing import Union
from fastapi import FastAPI
from detect import classification as dcl

app = FastAPI()


@app.get("/api/")
def api(plant_id: int, image_url: str):
    result = dcl.detect(plant_id, image_url)
	
    return {"item_id": result}
