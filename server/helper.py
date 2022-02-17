import os
import copy
import json
from shapely.geometry import Polygon
from pydantic import BaseModel

from yolov3_tf2 import yolov3_model
from mask_rcnn import maskrcnn_model

data = {}

yolo = yolov3_model.YoloV3Model(
    i_classes='./config/coco.names',
    i_yolo_max_boxes=100
)

mask_rcnn = maskrcnn_model.MaskRCNNModel(
    i_classes='./config/coco.names',
    i_weights='./mask_rcnn/model/mask_rcnn_coco.h5',
    i_logs='./mask_rcnn/logs/'
)

class GetBoundingBoxesRequest(BaseModel):
    image_id: str
    image_file_name: str

class GetObjectBoundaryRequest(BaseModel):
    image_id: str
    image_file_name: str
    bounding_box: list
    class_of_interest: int

class AnnotationResult(BaseModel):
    object_class: int
    predicted_bounding_box: list
    predicted_polygon: list
    annotated_bounding_box: list
    annotated_polygon: list
    bounding_box_changes: int
    polygon_changes: int

class SubmitResultRequest(BaseModel):
    image_id: str
    image_file_name: str
    result: AnnotationResult

class GetPolygonMetricsRequest(BaseModel):
    image_id: str
    predicted_polygon: list
    ground_truth_polygon: list

class GetBoundingBoxMetricsRequest(BaseModel):
    image_id: str
    predicted_bounding_box: list
    ground_truth_bounding_box: list

def get_bounding_boxes_helper(req: GetBoundingBoxesRequest):
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

    return image_path, npboxes, classes

def get_object_boundary_helper(req: GetObjectBoundaryRequest):
    image_path = "./data/" + req.image_file_name
    full_mask, simple_mask_polygon = mask_rcnn.predict(image_path, req.bounding_box, req.class_of_interest)

    return image_path, full_mask, simple_mask_polygon

def submit_result_helper(req: SubmitResultRequest):
    global data
    load_state_helper()
    if req.result is not None:
        if req.result.object_class in data:
            class_data = data.get(req.result.object_class)
        else:
            class_data = {}
            data[req.result.object_class] = class_data

        if req.image_file_name in class_data:
            image_data = data.get(req.image_file_name)
        else:
            image_data = {}
            data[req.image_file_name] = image_data

        image_data["image_id"] = req.image_id
        image_data["predicted_bounding_box"] = req.result.predicted_bounding_box
        image_data["predicted_polygon"] = req.result.predicted_polygon
        image_data["annotated_bounding_box"] = req.result.annotated_bounding_box
        image_data["annotated_polygon"] = req.result.annotated_polygon

        # TODO: read ground truth when available
        if True:
            image_data["ground_truth_bounding_box"] = [10, 10, 30, 30]
            image_data["ground_truth_polygon"] = [[10, 10], [10, 20], [10, 30], [20, 30], [30, 30], [30, 20], [30, 10], [20, 10]]
        else:
            image_data["ground_truth_bounding_box"] = None
            image_data["ground_truth_polygon"] = None

        image_data["metrics"] = {"an_vs_gt": {}, "pd_vs_gt": {}, "an_vs_pd": {}}
        image_metrics = image_data.get("metrics")

        # pd_vs_gt
        if image_data.get("predicted_bounding_box") is not None and image_data.get("ground_truth_bounding_box") is not None:
            pd_vs_gt = image_metrics.get("pd_vs_gt")
            r = GetBoundingBoxMetricsRequest(image_id=image_data.get("image_id"), \
                predicted_bounding_box=image_data.get("predicted_bounding_box"), \
                ground_truth_bounding_box=image_data.get("ground_truth_bounding_box"))
            pd_vs_gt["bb_iou"] = get_bounding_box_iou_helper(r)
            pd_vs_gt["bb_percentage_area_change"] = get_bounding_box_percentage_area_change_helper(r)
            pd_vs_gt["bb_number_of_changes"] = get_bounding_box_number_of_changes_helper(r)

        if image_data.get("predicted_polygon") is not None and image_data.get("ground_truth_polygon") is not None:
            pd_vs_gt = image_metrics.get("pd_vs_gt")
            r = GetPolygonMetricsRequest(image_id=image_data.get("image_id"), \
                predicted_polygon=image_data.get("predicted_polygon"), \
                ground_truth_polygon=image_data.get("ground_truth_polygon"))
            pd_vs_gt["p_iou"] = get_polygon_iou_helper(r)
            pd_vs_gt["p_percentage_area_change"] = get_polygon_percentage_area_change_helper(r)
            pd_vs_gt["p_number_of_changes"] = get_polygon_number_of_changes_helper(r)

        # an_vs_gt
        if image_data.get("annotated_bounding_box") is not None and image_data.get("ground_truth_bounding_box") is not None:
            an_vs_gt = image_metrics.get("an_vs_gt")
            r = GetBoundingBoxMetricsRequest(image_id=image_data.get("image_id"), \
                predicted_bounding_box=image_data.get("annotated_bounding_box"), \
                ground_truth_bounding_box=image_data.get("ground_truth_bounding_box"))
            an_vs_gt["bb_iou"] = get_bounding_box_iou_helper(r)
            an_vs_gt["bb_percentage_area_change"] = get_bounding_box_percentage_area_change_helper(r)
            an_vs_gt["bb_number_of_changes"] = get_bounding_box_number_of_changes_helper(r)

        if image_data.get("annotated_polygon") is not None and image_data.get("ground_truth_polygon") is not None:
            an_vs_gt = image_metrics.get("an_vs_gt")
            r = GetPolygonMetricsRequest(image_id=image_data.get("image_id"), \
                predicted_polygon=image_data.get("annotated_polygon"), \
                ground_truth_polygon=image_data.get("ground_truth_polygon"))
            an_vs_gt["p_iou"] = get_polygon_iou_helper(r)
            an_vs_gt["p_percentage_area_change"] = get_polygon_percentage_area_change_helper(r)
            an_vs_gt["p_number_of_changes"] = get_polygon_number_of_changes_helper(r)

        # an_vs_pd
        if image_data.get("annotated_bounding_box") is not None and image_data.get("predicted_bounding_box") is not None:
            an_vs_pd = image_metrics.get("an_vs_pd")
            r = GetBoundingBoxMetricsRequest(image_id=image_data.get("image_id"), \
                predicted_bounding_box=image_data.get("predicted_bounding_box"), \
                ground_truth_bounding_box=image_data.get("annotated_bounding_box"))
            an_vs_pd["bb_iou"] = get_bounding_box_iou_helper(r)
            an_vs_pd["bb_percentage_area_change"] = get_bounding_box_percentage_area_change_helper(r)
            an_vs_pd["bb_number_of_changes"] = req.result.bounding_box_changes

        if image_data.get("annotated_polygon") is not None and image_data.get("predicted_polygon") is not None:
            an_vs_pd = image_metrics.get("an_vs_pd")
            r = GetPolygonMetricsRequest(image_id=image_data.get("image_id"), \
                predicted_polygon=image_data.get("predicted_polygon"), \
                ground_truth_polygon=image_data.get("annotated_polygon"))
            an_vs_pd["p_iou"] = get_polygon_iou_helper(r)
            an_vs_pd["p_percentage_area_change"] = get_polygon_percentage_area_change_helper(r)
            an_vs_pd["p_number_of_changes"] = req.result.polygon_changes

    save_state_helper()

def get_polygon_iou_helper(req: GetPolygonMetricsRequest):
    ground_truth_points = [tuple(x) for x in req.ground_truth_polygon]
    predicted_points = [tuple(x) for x in req.predicted_polygon]
    ground_truth_polygon = Polygon(ground_truth_points)
    predicted_polygon = Polygon(predicted_points)
    iou = ground_truth_polygon.intersection(predicted_polygon).area / (ground_truth_polygon.union(predicted_polygon).area + 0.001)

    return iou

def get_bounding_box_iou_helper(req: GetBoundingBoxMetricsRequest):
    gt_tl = [req.ground_truth_bounding_box[0], req.ground_truth_bounding_box[1]]
    gt_br = [req.ground_truth_bounding_box[2], req.ground_truth_bounding_box[3]]
    ground_truth_points = [(gt_tl[0], gt_tl[1]), (gt_br[0], gt_tl[1]), (gt_br[0], gt_br[1]), (gt_tl[0], gt_br[1])]
    p_tl = [req.predicted_bounding_box[0], req.predicted_bounding_box[1]]
    p_br = [req.predicted_bounding_box[2], req.predicted_bounding_box[3]]
    predicted_points = [(p_tl[0], p_tl[1]), (p_br[0], p_tl[1]), (p_br[0], p_br[1]), (p_tl[0], p_br[1])]

    ground_truth_polygon = Polygon(ground_truth_points)
    predicted_polygon = Polygon(predicted_points)
    iou = ground_truth_polygon.intersection(predicted_polygon).area / (ground_truth_polygon.union(predicted_polygon).area + 0.001)

    return iou

def get_polygon_number_of_changes_helper(req: GetPolygonMetricsRequest):
    gtp = copy.deepcopy(req.ground_truth_polygon)
    pp = copy.deepcopy(req.predicted_polygon)
    count = 0
    for p in req.predicted_polygon:
        if p in req.ground_truth_polygon:
            gtp.remove(p)
            pp.remove(p)

    count = max(len(gtp), len(pp))

    return count

def get_bounding_box_number_of_changes_helper(req: GetPolygonMetricsRequest):
    gt_tl = [req.ground_truth_bounding_box[0], req.ground_truth_bounding_box[1]]
    gt_br = [req.ground_truth_bounding_box[2], req.ground_truth_bounding_box[3]]
    gtp = [[gt_tl[0], gt_tl[1]], [gt_br[0], gt_tl[1]], [gt_br[0], gt_br[1]], [gt_tl[0], gt_br[1]]]
    p_tl = [req.predicted_bounding_box[0], req.predicted_bounding_box[1]]
    p_br = [req.predicted_bounding_box[2], req.predicted_bounding_box[3]]
    pp = [[p_tl[0], p_tl[1]], [p_br[0], p_tl[1]], [p_br[0], p_br[1]], [p_tl[0], p_br[1]]]
    
    count = 0
    for p in copy.deepcopy(pp):
        if p in copy.deepcopy(gtp):
            gtp.remove(p)
            pp.remove(p)

    count = max(len(gtp), len(pp))

    return count

def get_polygon_percentage_area_change_helper(req: GetPolygonMetricsRequest):
    ground_truth_points = [tuple(x) for x in req.ground_truth_polygon]
    predicted_points = [tuple(x) for x in req.predicted_polygon]
    ground_truth_polygon = Polygon(ground_truth_points)
    predicted_polygon = Polygon(predicted_points)
    percentage_area_change = abs(ground_truth_polygon.area - predicted_polygon.area) / ground_truth_polygon.area

    return percentage_area_change

def get_bounding_box_percentage_area_change_helper(req: GetBoundingBoxMetricsRequest):
    gt_tl = [req.ground_truth_bounding_box[0], req.ground_truth_bounding_box[1]]
    gt_br = [req.ground_truth_bounding_box[2], req.ground_truth_bounding_box[3]]
    ground_truth_points = [(gt_tl[0], gt_tl[1]), (gt_br[0], gt_tl[1]), (gt_br[0], gt_br[1]), (gt_tl[0], gt_br[1])]
    p_tl = [req.predicted_bounding_box[0], req.predicted_bounding_box[1]]
    p_br = [req.predicted_bounding_box[2], req.predicted_bounding_box[3]]
    predicted_points = [(p_tl[0], p_tl[1]), (p_br[0], p_tl[1]), (p_br[0], p_br[1]), (p_tl[0], p_br[1])]

    ground_truth_polygon = Polygon(ground_truth_points)
    predicted_polygon = Polygon(predicted_points)
    percentage_area_change = abs(ground_truth_polygon.area - predicted_polygon.area) / ground_truth_polygon.area

    return percentage_area_change

def load_state_helper():
    global data
    if not data and os.path.exists("./data.json"):
        with open("./data.json") as json_file:
            data = json.load(json_file)

def save_state_helper():
    global data
    with open("./data.json", "w") as json_file:
        json.dump(data, json_file)