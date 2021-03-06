import time

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from os.path import isfile
from fastapi import Response
from mimetypes import guess_type

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
    submit_result_helper,
    compute_statistics_helper,
    recalculate_metrics_helper
)

app = FastAPI()

app.mount("/css", StaticFiles(directory="static/css"), name="static-css")
app.mount("/scripts", StaticFiles(directory="static/scripts"), name="static-scripts")

@app.get("/favicon.ico")
async def favicon():
    filename = './static/favicon.ico'

    if not isfile(filename):
        return Response(status_code=404)

    with open(filename, 'rb') as f:
        content = f.read()

    content_type, _ = guess_type(filename)
    return Response(content, media_type=content_type)

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

@app.post('/recalculate_metrics')
def recalculate_metrics():
    recalculate_metrics_helper()

    return {
        'status': 'Success',
    }

@app.post('/compute_statistics')
def compute_statistics():
    compute_statistics_helper()

    return {
        'status': 'Success',
    }

@app.post('/get_polygon_iou')
def get_polygon_IOU(req: GetPolygonMetricsRequest):
    iou = get_polygon_iou_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'iou': iou
    }

@app.post('/get_bounding_box_iou')
def get_bounding_box_IOU(req: GetBoundingBoxMetricsRequest):
    iou = get_bounding_box_iou_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'iou': iou
    }

@app.post('/get_polygon_number_of_changes')
def get_polygon_number_of_changes(req: GetPolygonMetricsRequest):
    count = get_polygon_number_of_changes_helper(req)

    return {
        'message': f'image id: {req.image_id}',
        'changes': count
    }

@app.post('/get_bounding_box_number_of_changes')
def get_bounding_box_number_of_changes(req: GetBoundingBoxMetricsRequest):
    count = get_bounding_box_number_of_changes_helper(req)

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

@app.post('/upload_image')
async def upload_image(file: UploadFile = File(...)):
    print(file)

    content = file.file.read()

    # save file
    fn = 'upload_{}_{}'.format(int(time.time()), file.filename)
    with open(f'./data/{fn}', 'wb') as f:
        f.write(content)

    return {
        'filename': fn
    }
