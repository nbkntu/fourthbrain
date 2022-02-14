import copy
from shapely.geometry import Polygon
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
    print(req)

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
    for p in pp:
        if p not in gtp:
            count += 1
        else:
            gtp.remove(p)

    count += len(gtp)

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