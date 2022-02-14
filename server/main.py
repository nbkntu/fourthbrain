from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from helper import (
    GetBoundingBoxesRequest,
    GetObjectBoundaryRequest,
    GetPolygonMetricsRequest,
    GetBoundingBoxMetricsRequest,
    SubmitResultRequest,
    get_bounding_boxes_helper,
    get_bounding_box_iou_helper,
    get_bounding_box_percentage_area_change_helper,
    get_object_boundary_helper,
    get_polygon_iou_helper,
    get_polygon_number_of_changes_helper,
    get_polygon_percentage_area_change_helper,
    submit_result_helper
)

app = FastAPI()

app.mount("/css", StaticFiles(directory="static/css"), name="static-css")
app.mount("/scripts", StaticFiles(directory="static/scripts"), name="static-scripts")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/get_bounding_boxes')
def get_bounding_boxes(req: GetBoundingBoxesRequest):
    image_path, npboxes, classes = get_bounding_boxes_helper(req)

    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'bounding_box': npboxes.tolist(),
        'classes': classes.tolist()
    }

@app.post('/get_object_boundary')
def get_object_boundary(req: GetObjectBoundaryRequest):
    image_path, _, simple_mask_polygon = get_object_boundary_helper(req)

    return {
        'message': f'image id: {req.image_id}, image path: {image_path}',
        'simple_mask_polygon': simple_mask_polygon.tolist()
    }

@app.post('/submit_result')
def submit_result(req: SubmitResultRequest):
    submit_result_helper(req)

    return {
        'status': 'Success',
    }

@app.post('/get_polygon_IOU')
def get_polygon_IOU(req: GetPolygonMetricsRequest):
    IOU = get_polygon_iou_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'IOU': IOU
    }

@app.post('/get_bounding_box_IOU')
def get_bounding_box_IOU(req: GetBoundingBoxMetricsRequest):
    IOU = get_bounding_box_iou_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'IOU': IOU
    }

@app.post('/get_polygon_number_of_changes')
def get_polygon_number_of_changes(req: GetPolygonMetricsRequest):
    count = get_polygon_number_of_changes_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'changes': count
    }

@app.post('/get_polygon_percentage_area_change')
def get_polygon_percentage_area_change(req: GetPolygonMetricsRequest):
    percentage_area_change = get_polygon_percentage_area_change_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'percentage_area_change': percentage_area_change
    }

@app.post('/get_bounding_box_percentage_area_change')
def get_bounding_box_percentage_area_change(req: GetBoundingBoxMetricsRequest):
    percentage_area_change = get_bounding_box_percentage_area_change_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'percentage_area_change': percentage_area_change
    }