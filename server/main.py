import uvicorn
import subprocess
import os
from shapely.geometry import Polygon

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from yolov3_tf2 import yolov3_model
from mask_rcnn import maskrcnn_model

yolo = yolov3_model.YoloV3Model(
    i_classes='./config/coco.names',
    i_yolo_max_boxes=10
)

mask_rcnn = maskrcnn_model.MaskRCNNModel(
    i_classes='./config/coco.names',
    i_weights='./mask_rcnn/model/mask_rcnn_coco.h5',
    i_logs='./mask_rcnn/logs/'
)

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

class GetObjectBoundaryRequest(BaseModel):
    image_id: str
    image_file_name: str
    bounding_box: list
    class_of_interest: int

class GetPolygonIOURequest(BaseModel):
    image_id: str
    predicted_polygon: list
    ground_truth_polygon: list

class GetBoundingBoxIOURequest(BaseModel):
    image_id: str
    predicted_bounding_box: list
    ground_truth_bounding_box: list

@app.post('/get_bounding_boxes')
def get_bounding_boxes(req: GetBoundingBoxesRequest):
    image_path = "./data/" + req.image_file_name
    # img_shape: (height, width, channels), box point: (width, height), box: [top_left_box_point bottom_right_box_point]
    boxes, _, classes, _, img_shape = yolo.process(req.image_id, image_path)

    npboxes = boxes.numpy()
    npboxes = npboxes.reshape((npboxes.shape[1], npboxes.shape[2]))
    for i in range(npboxes.shape[0]):
        npboxes[i][0] = max(0, int(npboxes[i][0]*img_shape[1]))
        npboxes[i][1] = max(0, int(npboxes[i][1]*img_shape[0]))
        npboxes[i][2] = max(0, int(npboxes[i][2]*img_shape[1]))
        npboxes[i][3] = max(0, int(npboxes[i][3]*img_shape[0]))
    
    npboxes, classes = yolo.non_maximum_suppression(npboxes, classes.numpy().flatten())

    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'bounding_box': npboxes.tolist(),
        'classes': classes.tolist()
    }

@app.post('/get_object_boundary')
def get_object_boundary(req: GetObjectBoundaryRequest):
    image_path = "./data/" + req.image_file_name
    full_mask, simple_mask_polygon = mask_rcnn.predict(image_path, req.bounding_box, req.class_of_interest)

    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'simple_mask_polygon': simple_mask_polygon.tolist()
    }

@app.post('/get_polygon_IOU')
def get_polygon_IOU(req: GetPolygonIOURequest):
    ground_truth_points = [tuple(x) for x in req.ground_truth_polygon]
    predicted_points = [tuple(x) for x in req.predicted_polygon]
    ground_truth_polygon = Polygon(ground_truth_points)
    predicted_polygon = Polygon(predicted_points)
    IOU = ground_truth_polygon.intersection(predicted_polygon).area / (ground_truth_polygon.union(predicted_polygon).area + 0.001)

    return {
        'message': f'image id: {req.image_id}',
        'IOU': IOU
    }

@app.post('/get_bounding_box_IOU')
def get_bounding_box_IOU(req: GetBoundingBoxIOURequest):
    gt_tl = [req.ground_truth_bounding_box[0], req.ground_truth_bounding_box[1]]
    gt_br = [req.ground_truth_bounding_box[2], req.ground_truth_bounding_box[3]]
    ground_truth_points = [(gt_tl[0], gt_tl[1]), (gt_br[0], gt_tl[1]), (gt_br[0], gt_br[1]), (gt_tl[0], gt_br[1])]
    p_tl = [req.predicted_bounding_box[0], req.predicted_bounding_box[1]]
    p_br = [req.predicted_bounding_box[2], req.predicted_bounding_box[3]]
    predicted_points = [(p_tl[0], p_tl[1]), (p_br[0], p_tl[1]), (p_br[0], p_br[1]), (p_tl[0], p_br[1])]
    
    ground_truth_polygon = Polygon(ground_truth_points)
    predicted_polygon = Polygon(predicted_points)
    IOU = ground_truth_polygon.intersection(predicted_polygon).area / (ground_truth_polygon.union(predicted_polygon).area + 0.001)

    return {
        'message': f'image id: {req.image_id}',
        'IOU': IOU
    }