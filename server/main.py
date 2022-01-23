import uvicorn
import subprocess
import os

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from yolov3_tf2 import detect2



app = FastAPI()

app.mount("/css", StaticFiles(directory="static/css"), name="static-css")
app.mount("/scripts", StaticFiles(directory="static/scripts"), name="static-scripts")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class GetBoundingBoxesRequest(BaseModel):
    image_id: str
    image_file_name: str

@app.post('/get_bounding_boxes')
def get_bounding_boxes(req: GetBoundingBoxesRequest):
    image_path = "./yolov3_tf2/data/" + req.image_file_name
    boxes, scores, classes, nums, img_shape = detect2.detect(req.image_id, image_path, i_classes='./yolov3_tf2/data/coco2.names', i_yolo_max_boxes=1)
    npboxes = boxes.numpy()
    npboxes = npboxes.reshape((npboxes.shape[1], npboxes.shape[2]))
    for i in range(npboxes.shape[0]):
        npboxes[i][0] = npboxes[i][0]*img_shape[1]
        npboxes[i][1] = npboxes[i][1]*img_shape[0]
        npboxes[i][2] = npboxes[i][2]*img_shape[1]
        npboxes[i][3] = npboxes[i][3]*img_shape[0]
    
    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'bounding_box': npboxes.tolist(),
        'classes': classes.numpy().flatten().tolist()
    }
