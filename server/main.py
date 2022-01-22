from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import subprocess
import os
import detect2

from pydantic import BaseModel

app = FastAPI()

app.mount("/css", StaticFiles(directory="static/css"), name="static-css")
app.mount("/scripts", StaticFiles(directory="static/scripts"), name="static-scripts")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class HelloRequest(BaseModel):
    name: str

@app.post("/hello")
async def sayHello(hello_request: HelloRequest):
    return {"message": f"Hello, {hello_request.name}!"}

@app.get('/{image_id}/{image_file_name}')
def get_bounding_boxes(image_id: str, image_file_name: str):
    image_path = "./yolov3_tf2/data/" + image_file_name
    boxes, scores, classes, nums, img_shape = detect2.detect(image_id, image_path, i_classes='./yolov3_tf2/data/coco2.names', i_yolo_max_boxes=1)
    npboxes = boxes.numpy()
    npboxes = npboxes.reshape((npboxes.shape[1], npboxes.shape[2]))
    for i in range(npboxes.shape[0]):
        npboxes[i][0] = npboxes[i][0]*img_shape[1]
        npboxes[i][1] = npboxes[i][1]*img_shape[0]
        npboxes[i][2] = npboxes[i][2]*img_shape[1]
        npboxes[i][3] = npboxes[i][3]*img_shape[0]
    
    return {'message': f'image id: {image_id}, image path: {image_path}',
            'bounding_box': npboxes.tolist(),
            'classes': classes.numpy().flatten().tolist()
        }
