import uvicorn
import subprocess
import os

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from yolov3_tf2 import yolov3_model
from mask_rcnn import maskrcnn_model


app = FastAPI()

yolo = yolov3_model.YoloV3Model(
    i_classes='./config/coco.names',
    i_yolo_max_boxes=10
)

mask_rcnn = maskrcnn_model.MaskRCNNModel(
    i_classes='./config/coco.names',
    i_weights='./mask_rcnn/model/mask_rcnn_coco.h5',
    i_logs='./mask_rcnn/logs/'
)

app.mount("/css", StaticFiles(directory="static/css"), name="static-css")
app.mount("/scripts", StaticFiles(directory="static/scripts"), name="static-scripts")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class GetBoundingBoxesRequest(BaseModel):
    image_id: str
    image_file_name: str

class GetObjectBoundaryRequest(BaseModel):
    image_id: str
    image_file_name: str
    bounding_box: list
    class_of_interest: int

@app.post('/get_bounding_boxes')
def get_bounding_boxes(req: GetBoundingBoxesRequest):
    image_path = "./data/" + req.image_file_name
    boxes, scores, classes, nums, img_shape = yolo.process(req.image_id, image_path)

    npboxes = boxes.numpy()
    npboxes = npboxes.reshape((npboxes.shape[1], npboxes.shape[2]))
    for i in range(npboxes.shape[0]):
        npboxes[i][0] = int(npboxes[i][0]*img_shape[1])
        npboxes[i][1] = int(npboxes[i][1]*img_shape[0])
        npboxes[i][2] = int(npboxes[i][2]*img_shape[1])
        npboxes[i][3] = int(npboxes[i][3]*img_shape[0])

    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'bounding_box': npboxes.tolist(),
        'classes': classes.numpy().flatten().tolist()
    }

@app.post('/get_object_boundary')
def get_object_boundary(req: GetObjectBoundaryRequest):
    image_path = "./data/" + req.image_file_name
    full_mask, simple_mask_polygon = mask_rcnn.predict(image_path, req.bounding_box, req.class_of_interest)

    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'simple_mask_polygon': simple_mask_polygon.tolist()
    }
