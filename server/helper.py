import os
import copy
import json
from shapely.geometry import Polygon
from pydantic import BaseModel

from coco.cocotools import CocoUtils
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

coco_utils = CocoUtils()

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

def get_points(segmentation):
    s = segmentation[0]
    points = []
    for i in range(0, len(s), 2):
        points.append([int(s[i]), int(s[i+1])])

    return points

def find_best_ground_truth(gts, ann_bbox):
    first = True
    ground_truth_bounding_box = None
    ground_truth_polygon = None
    mx = 0
    for g in gts:
        if first:
            first = False
            ground_truth_bounding_box = [int(x) for x in g.get("bbox")]
            ground_truth_polygon = get_points(g.get("segmentation"))
            mx = get_bounding_box_iou_helper(GetBoundingBoxMetricsRequest(image_id="", \
                                                                            predicted_bounding_box=ann_bbox, \
                                                                            ground_truth_bounding_box=ground_truth_bounding_box))
        else:
            bbox = [int(x) for x in g.get("bbox")]
            m = get_bounding_box_iou_helper(GetBoundingBoxMetricsRequest(image_id="", \
                                                                            predicted_bounding_box=ann_bbox, \
                                                                            ground_truth_bounding_box=bbox))
            if (m > mx):
                mx = m
                ground_truth_bounding_box = bbox
                ground_truth_polygon = get_points(g.get("segmentation"))


    return ground_truth_bounding_box, ground_truth_polygon

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
            image_data = class_data.get(req.image_file_name)
        else:
            image_data = {}
            class_data[req.image_file_name] = image_data

        image_data["image_id"] = req.image_id
        image_data["predicted_bounding_box"] = req.result.predicted_bounding_box
        image_data["predicted_polygon"] = req.result.predicted_polygon
        image_data["annotated_bounding_box"] = req.result.annotated_bounding_box
        image_data["annotated_polygon"] = req.result.annotated_polygon


        image_data["ground_truth_bounding_box"] = None
        image_data["ground_truth_polygon"] = None
        file_name = os.path.splitext(req.image_file_name)[0]
        # coco image id is a number
        if file_name.isnumeric():
            gts = coco_utils.load_annotations(int(file_name), yolo.get_class_name(req.result.object_class))
            ground_truth_bounding_box, ground_truth_polygon = find_best_ground_truth(gts, req.result.annotated_bounding_box)
            if ground_truth_bounding_box is not None:
                image_data["ground_truth_bounding_box"] = ground_truth_bounding_box
                image_data["ground_truth_polygon"] = ground_truth_polygon

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

def recalculate_metrics_helper():
    global data
    load_state_helper()

    for _, class_data in data.items():
        for _, image_data in class_data.items():
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

            if image_data.get("annotated_polygon") is not None and image_data.get("predicted_polygon") is not None:
                an_vs_pd = image_metrics.get("an_vs_pd")
                r = GetPolygonMetricsRequest(image_id=image_data.get("image_id"), \
                    predicted_polygon=image_data.get("predicted_polygon"), \
                    ground_truth_polygon=image_data.get("annotated_polygon"))
                an_vs_pd["p_iou"] = get_polygon_iou_helper(r)
                an_vs_pd["p_percentage_area_change"] = get_polygon_percentage_area_change_helper(r)

    save_state_helper()

def compute_statistics_helper():
    global data
    load_state_helper()

    overall_pd_vs_gt_bb_min_iou = 2
    overall_pd_vs_gt_bb_max_iou = -1
    overall_pd_vs_gt_bb_sum_iou = 0
    overall_pd_vs_gt_bb_min_percentage_area_change = 200
    overall_pd_vs_gt_bb_max_percentage_area_change = -1
    overall_pd_vs_gt_bb_sum_percentage_area_change = 0
    overall_pd_vs_gt_bb_min_number_of_changes = 10000
    overall_pd_vs_gt_bb_max_number_of_changes = -1
    overall_pd_vs_gt_bb_sum_number_of_changes = 0
    overall_pd_vs_gt_p_min_iou = 2
    overall_pd_vs_gt_p_max_iou = -1
    overall_pd_vs_gt_p_sum_iou = 0
    overall_pd_vs_gt_p_min_percentage_area_change = 200
    overall_pd_vs_gt_p_max_percentage_area_change = -1
    overall_pd_vs_gt_p_sum_percentage_area_change = 0
    overall_pd_vs_gt_p_min_number_of_changes = 10000
    overall_pd_vs_gt_p_max_number_of_changes = -1
    overall_pd_vs_gt_p_sum_number_of_changes = 0

    overall_an_vs_gt_bb_min_iou = 2
    overall_an_vs_gt_bb_max_iou = -1
    overall_an_vs_gt_bb_sum_iou = 0
    overall_an_vs_gt_bb_min_percentage_area_change = 200
    overall_an_vs_gt_bb_max_percentage_area_change = -1
    overall_an_vs_gt_bb_sum_percentage_area_change = 0
    overall_an_vs_gt_bb_min_number_of_changes = 10000
    overall_an_vs_gt_bb_max_number_of_changes = -1
    overall_an_vs_gt_bb_sum_number_of_changes = 0
    overall_an_vs_gt_p_min_iou = 2
    overall_an_vs_gt_p_max_iou = -1
    overall_an_vs_gt_p_sum_iou = 0
    overall_an_vs_gt_p_min_percentage_area_change = 200
    overall_an_vs_gt_p_max_percentage_area_change = -1
    overall_an_vs_gt_p_sum_percentage_area_change = 0
    overall_an_vs_gt_p_min_number_of_changes = 10000
    overall_an_vs_gt_p_max_number_of_changes = -1
    overall_an_vs_gt_p_sum_number_of_changes = 0

    overall_an_vs_pd_bb_min_iou = 2
    overall_an_vs_pd_bb_max_iou = -1
    overall_an_vs_pd_bb_sum_iou = 0
    overall_an_vs_pd_bb_min_percentage_area_change = 200
    overall_an_vs_pd_bb_max_percentage_area_change = -1
    overall_an_vs_pd_bb_sum_percentage_area_change = 0
    overall_an_vs_pd_bb_min_number_of_changes = 10000
    overall_an_vs_pd_bb_max_number_of_changes = -1
    overall_an_vs_pd_bb_sum_number_of_changes = 0
    overall_an_vs_pd_p_min_iou = 2
    overall_an_vs_pd_p_max_iou = -1
    overall_an_vs_pd_p_sum_iou = 0
    overall_an_vs_pd_p_min_percentage_area_change = 200
    overall_an_vs_pd_p_max_percentage_area_change = -1
    overall_an_vs_pd_p_sum_percentage_area_change = 0
    overall_an_vs_pd_p_min_number_of_changes = 10000
    overall_an_vs_pd_p_max_number_of_changes = -1
    overall_an_vs_pd_p_sum_number_of_changes = 0
    overall_count = 0

    for class_id, class_data in data.items():
        pd_vs_gt_bb_min_iou = 2
        pd_vs_gt_bb_max_iou = -1
        pd_vs_gt_bb_sum_iou = 0
        pd_vs_gt_bb_min_percentage_area_change = 200
        pd_vs_gt_bb_max_percentage_area_change = -1
        pd_vs_gt_bb_sum_percentage_area_change = 0
        pd_vs_gt_bb_min_number_of_changes = 10000
        pd_vs_gt_bb_max_number_of_changes = -1
        pd_vs_gt_bb_sum_number_of_changes = 0
        pd_vs_gt_p_min_iou = 2
        pd_vs_gt_p_max_iou = -1
        pd_vs_gt_p_sum_iou = 0
        pd_vs_gt_p_min_percentage_area_change = 200
        pd_vs_gt_p_max_percentage_area_change = -1
        pd_vs_gt_p_sum_percentage_area_change = 0
        pd_vs_gt_p_min_number_of_changes = 10000
        pd_vs_gt_p_max_number_of_changes = -1
        pd_vs_gt_p_sum_number_of_changes = 0

        an_vs_gt_bb_min_iou = 2
        an_vs_gt_bb_max_iou = -1
        an_vs_gt_bb_sum_iou = 0
        an_vs_gt_bb_min_percentage_area_change = 200
        an_vs_gt_bb_max_percentage_area_change = -1
        an_vs_gt_bb_sum_percentage_area_change = 0
        an_vs_gt_bb_min_number_of_changes = 10000
        an_vs_gt_bb_max_number_of_changes = -1
        an_vs_gt_bb_sum_number_of_changes = 0
        an_vs_gt_p_min_iou = 2
        an_vs_gt_p_max_iou = -1
        an_vs_gt_p_sum_iou = 0
        an_vs_gt_p_min_percentage_area_change = 200
        an_vs_gt_p_max_percentage_area_change = -1
        an_vs_gt_p_sum_percentage_area_change = 0
        an_vs_gt_p_min_number_of_changes = 10000
        an_vs_gt_p_max_number_of_changes = -1
        an_vs_gt_p_sum_number_of_changes = 0

        an_vs_pd_bb_min_iou = 2
        an_vs_pd_bb_max_iou = -1
        an_vs_pd_bb_sum_iou = 0
        an_vs_pd_bb_min_percentage_area_change = 200
        an_vs_pd_bb_max_percentage_area_change = -1
        an_vs_pd_bb_sum_percentage_area_change = 0
        an_vs_pd_bb_min_number_of_changes = 10000
        an_vs_pd_bb_max_number_of_changes = -1
        an_vs_pd_bb_sum_number_of_changes = 0
        an_vs_pd_p_min_iou = 2
        an_vs_pd_p_max_iou = -1
        an_vs_pd_p_sum_iou = 0
        an_vs_pd_p_min_percentage_area_change = 200
        an_vs_pd_p_max_percentage_area_change = -1
        an_vs_pd_p_sum_percentage_area_change = 0
        an_vs_pd_p_min_number_of_changes = 10000
        an_vs_pd_p_max_number_of_changes = -1
        an_vs_pd_p_sum_number_of_changes = 0
        count = 0

        for _, image_data in class_data.items():
            count += 1
            overall_count += 1
            metrics = image_data.get("metrics")
            pd_vs_gt = metrics.get("pd_vs_gt")
            pd_vs_gt_bb_min_iou = min(pd_vs_gt.get("bb_iou"), pd_vs_gt_bb_min_iou)
            pd_vs_gt_bb_max_iou = max(pd_vs_gt.get("bb_iou"), pd_vs_gt_bb_max_iou)
            pd_vs_gt_bb_sum_iou += pd_vs_gt.get("bb_iou")
            pd_vs_gt_bb_min_percentage_area_change = min(pd_vs_gt.get("bb_percentage_area_change"), pd_vs_gt_bb_min_percentage_area_change)
            pd_vs_gt_bb_max_percentage_area_change = max(pd_vs_gt.get("bb_percentage_area_change"), pd_vs_gt_bb_max_percentage_area_change)
            pd_vs_gt_bb_sum_percentage_area_change += pd_vs_gt.get("bb_percentage_area_change")
            pd_vs_gt_bb_min_number_of_changes = min(pd_vs_gt.get("bb_number_of_changes"), pd_vs_gt_bb_min_number_of_changes)
            pd_vs_gt_bb_max_number_of_changes = max(pd_vs_gt.get("bb_number_of_changes"), pd_vs_gt_bb_max_number_of_changes)
            pd_vs_gt_bb_sum_number_of_changes += pd_vs_gt.get("bb_number_of_changes")
            pd_vs_gt_p_min_iou = min(pd_vs_gt.get("p_iou"), pd_vs_gt_p_min_iou)
            pd_vs_gt_p_max_iou = max(pd_vs_gt.get("p_iou"), pd_vs_gt_p_max_iou)
            pd_vs_gt_p_sum_iou += pd_vs_gt.get("p_iou")
            pd_vs_gt_p_min_percentage_area_change = min(pd_vs_gt.get("p_percentage_area_change"), pd_vs_gt_p_min_percentage_area_change)
            pd_vs_gt_p_max_percentage_area_change = max(pd_vs_gt.get("p_percentage_area_change"), pd_vs_gt_p_max_percentage_area_change)
            pd_vs_gt_p_sum_percentage_area_change += pd_vs_gt.get("p_percentage_area_change")
            pd_vs_gt_p_min_number_of_changes = min(pd_vs_gt.get("p_number_of_changes"), pd_vs_gt_p_min_number_of_changes)
            pd_vs_gt_p_max_number_of_changes = max(pd_vs_gt.get("p_number_of_changes"), pd_vs_gt_p_max_number_of_changes)
            pd_vs_gt_p_sum_number_of_changes += pd_vs_gt.get("p_number_of_changes")

            overall_pd_vs_gt_bb_min_iou = min(pd_vs_gt.get("bb_iou"), overall_pd_vs_gt_bb_min_iou)
            overall_pd_vs_gt_bb_max_iou = max(pd_vs_gt.get("bb_iou"), overall_pd_vs_gt_bb_max_iou)
            overall_pd_vs_gt_bb_sum_iou += pd_vs_gt.get("bb_iou")
            overall_pd_vs_gt_bb_min_percentage_area_change = min(pd_vs_gt.get("bb_percentage_area_change"), overall_pd_vs_gt_bb_min_percentage_area_change)
            overall_pd_vs_gt_bb_max_percentage_area_change = max(pd_vs_gt.get("bb_percentage_area_change"), overall_pd_vs_gt_bb_max_percentage_area_change)
            overall_pd_vs_gt_bb_sum_percentage_area_change += pd_vs_gt.get("bb_percentage_area_change")
            overall_pd_vs_gt_bb_min_number_of_changes = min(pd_vs_gt.get("bb_number_of_changes"), overall_pd_vs_gt_bb_min_number_of_changes)
            overall_pd_vs_gt_bb_max_number_of_changes = max(pd_vs_gt.get("bb_number_of_changes"), overall_pd_vs_gt_bb_max_number_of_changes)
            overall_pd_vs_gt_bb_sum_number_of_changes += pd_vs_gt.get("bb_number_of_changes")
            overall_pd_vs_gt_p_min_iou = min(pd_vs_gt.get("p_iou"), overall_pd_vs_gt_p_min_iou)
            overall_pd_vs_gt_p_max_iou = max(pd_vs_gt.get("p_iou"), overall_pd_vs_gt_p_max_iou)
            overall_pd_vs_gt_p_sum_iou += pd_vs_gt.get("p_iou")
            overall_pd_vs_gt_p_min_percentage_area_change = min(pd_vs_gt.get("p_percentage_area_change"), overall_pd_vs_gt_p_min_percentage_area_change)
            overall_pd_vs_gt_p_max_percentage_area_change = max(pd_vs_gt.get("p_percentage_area_change"), overall_pd_vs_gt_p_max_percentage_area_change)
            overall_pd_vs_gt_p_sum_percentage_area_change += pd_vs_gt.get("p_percentage_area_change")
            overall_pd_vs_gt_p_min_number_of_changes = min(pd_vs_gt.get("p_number_of_changes"), overall_pd_vs_gt_p_min_number_of_changes)
            overall_pd_vs_gt_p_max_number_of_changes = max(pd_vs_gt.get("p_number_of_changes"), overall_pd_vs_gt_p_max_number_of_changes)
            overall_pd_vs_gt_p_sum_number_of_changes += pd_vs_gt.get("p_number_of_changes")

            an_vs_gt = metrics.get("an_vs_gt")
            an_vs_gt_bb_min_iou = min(an_vs_gt.get("bb_iou"), an_vs_gt_bb_min_iou)
            an_vs_gt_bb_max_iou = max(an_vs_gt.get("bb_iou"), an_vs_gt_bb_max_iou)
            an_vs_gt_bb_sum_iou += an_vs_gt.get("bb_iou")
            an_vs_gt_bb_min_percentage_area_change = min(an_vs_gt.get("bb_percentage_area_change"), an_vs_gt_bb_min_percentage_area_change)
            an_vs_gt_bb_max_percentage_area_change = max(an_vs_gt.get("bb_percentage_area_change"), an_vs_gt_bb_max_percentage_area_change)
            an_vs_gt_bb_sum_percentage_area_change += an_vs_gt.get("bb_percentage_area_change")
            an_vs_gt_bb_min_number_of_changes = min(an_vs_gt.get("bb_number_of_changes"), an_vs_gt_bb_min_number_of_changes)
            an_vs_gt_bb_max_number_of_changes = max(an_vs_gt.get("bb_number_of_changes"), an_vs_gt_bb_max_number_of_changes)
            an_vs_gt_bb_sum_number_of_changes += an_vs_gt.get("bb_number_of_changes")
            an_vs_gt_p_min_iou = min(an_vs_gt.get("p_iou"), an_vs_gt_p_min_iou)
            an_vs_gt_p_max_iou = max(an_vs_gt.get("p_iou"), an_vs_gt_p_max_iou)
            an_vs_gt_p_sum_iou += an_vs_gt.get("p_iou")
            an_vs_gt_p_min_percentage_area_change = min(an_vs_gt.get("p_percentage_area_change"), an_vs_gt_p_min_percentage_area_change)
            an_vs_gt_p_max_percentage_area_change = max(an_vs_gt.get("p_percentage_area_change"), an_vs_gt_p_max_percentage_area_change)
            an_vs_gt_p_sum_percentage_area_change += an_vs_gt.get("p_percentage_area_change")
            an_vs_gt_p_min_number_of_changes = min(an_vs_gt.get("p_number_of_changes"), an_vs_gt_p_min_number_of_changes)
            an_vs_gt_p_max_number_of_changes = max(an_vs_gt.get("p_number_of_changes"), an_vs_gt_p_max_number_of_changes)
            an_vs_gt_p_sum_number_of_changes += an_vs_gt.get("p_number_of_changes")

            overall_an_vs_gt_bb_min_iou = min(an_vs_gt.get("bb_iou"), overall_an_vs_gt_bb_min_iou)
            overall_an_vs_gt_bb_max_iou = max(an_vs_gt.get("bb_iou"), overall_an_vs_gt_bb_max_iou)
            overall_an_vs_gt_bb_sum_iou += an_vs_gt.get("bb_iou")
            overall_an_vs_gt_bb_min_percentage_area_change = min(an_vs_gt.get("bb_percentage_area_change"), overall_an_vs_gt_bb_min_percentage_area_change)
            overall_an_vs_gt_bb_max_percentage_area_change = max(an_vs_gt.get("bb_percentage_area_change"), overall_an_vs_gt_bb_max_percentage_area_change)
            overall_an_vs_gt_bb_sum_percentage_area_change += an_vs_gt.get("bb_percentage_area_change")
            overall_an_vs_gt_bb_min_number_of_changes = min(an_vs_gt.get("bb_number_of_changes"), overall_an_vs_gt_bb_min_number_of_changes)
            overall_an_vs_gt_bb_max_number_of_changes = max(an_vs_gt.get("bb_number_of_changes"), overall_an_vs_gt_bb_max_number_of_changes)
            overall_an_vs_gt_bb_sum_number_of_changes += an_vs_gt.get("bb_number_of_changes")
            overall_an_vs_gt_p_min_iou = min(an_vs_gt.get("p_iou"), overall_an_vs_gt_p_min_iou)
            overall_an_vs_gt_p_max_iou = max(an_vs_gt.get("p_iou"), overall_an_vs_gt_p_max_iou)
            overall_an_vs_gt_p_sum_iou += an_vs_gt.get("p_iou")
            overall_an_vs_gt_p_min_percentage_area_change = min(an_vs_gt.get("p_percentage_area_change"), overall_an_vs_gt_p_min_percentage_area_change)
            overall_an_vs_gt_p_max_percentage_area_change = max(an_vs_gt.get("p_percentage_area_change"), overall_an_vs_gt_p_max_percentage_area_change)
            overall_an_vs_gt_p_sum_percentage_area_change += an_vs_gt.get("p_percentage_area_change")
            overall_an_vs_gt_p_min_number_of_changes = min(an_vs_gt.get("p_number_of_changes"), overall_an_vs_gt_p_min_number_of_changes)
            overall_an_vs_gt_p_max_number_of_changes = max(an_vs_gt.get("p_number_of_changes"), overall_an_vs_gt_p_max_number_of_changes)
            overall_an_vs_gt_p_sum_number_of_changes += an_vs_gt.get("p_number_of_changes")

            an_vs_pd = metrics.get("an_vs_pd")
            an_vs_pd_bb_min_iou = min(an_vs_pd.get("bb_iou"), an_vs_pd_bb_min_iou)
            an_vs_pd_bb_max_iou = max(an_vs_pd.get("bb_iou"), an_vs_pd_bb_max_iou)
            an_vs_pd_bb_sum_iou += an_vs_pd.get("bb_iou")
            an_vs_pd_bb_min_percentage_area_change = min(an_vs_pd.get("bb_percentage_area_change"), an_vs_pd_bb_min_percentage_area_change)
            an_vs_pd_bb_max_percentage_area_change = max(an_vs_pd.get("bb_percentage_area_change"), an_vs_pd_bb_max_percentage_area_change)
            an_vs_pd_bb_sum_percentage_area_change += an_vs_pd.get("bb_percentage_area_change")
            an_vs_pd_bb_min_number_of_changes = min(an_vs_pd.get("bb_number_of_changes"), an_vs_pd_bb_min_number_of_changes)
            an_vs_pd_bb_max_number_of_changes = max(an_vs_pd.get("bb_number_of_changes"), an_vs_pd_bb_max_number_of_changes)
            an_vs_pd_bb_sum_number_of_changes += an_vs_pd.get("bb_number_of_changes")
            an_vs_pd_p_min_iou = min(an_vs_pd.get("p_iou"), an_vs_pd_p_min_iou)
            an_vs_pd_p_max_iou = max(an_vs_pd.get("p_iou"), an_vs_pd_p_max_iou)
            an_vs_pd_p_sum_iou += an_vs_pd.get("p_iou")
            an_vs_pd_p_min_percentage_area_change = min(an_vs_pd.get("p_percentage_area_change"), an_vs_pd_p_min_percentage_area_change)
            an_vs_pd_p_max_percentage_area_change = max(an_vs_pd.get("p_percentage_area_change"), an_vs_pd_p_max_percentage_area_change)
            an_vs_pd_p_sum_percentage_area_change += an_vs_pd.get("p_percentage_area_change")
            an_vs_pd_p_min_number_of_changes = min(an_vs_pd.get("p_number_of_changes"), an_vs_pd_p_min_number_of_changes)
            an_vs_pd_p_max_number_of_changes = max(an_vs_pd.get("p_number_of_changes"), an_vs_pd_p_max_number_of_changes)
            an_vs_pd_p_sum_number_of_changes += an_vs_pd.get("p_number_of_changes")

            overall_an_vs_pd_bb_min_iou = min(an_vs_pd.get("bb_iou"), overall_an_vs_pd_bb_min_iou)
            overall_an_vs_pd_bb_max_iou = max(an_vs_pd.get("bb_iou"), overall_an_vs_pd_bb_max_iou)
            overall_an_vs_pd_bb_sum_iou += an_vs_pd.get("bb_iou")
            overall_an_vs_pd_bb_min_percentage_area_change = min(an_vs_pd.get("bb_percentage_area_change"), overall_an_vs_pd_bb_min_percentage_area_change)
            overall_an_vs_pd_bb_max_percentage_area_change = max(an_vs_pd.get("bb_percentage_area_change"), overall_an_vs_pd_bb_max_percentage_area_change)
            overall_an_vs_pd_bb_sum_percentage_area_change += an_vs_pd.get("bb_percentage_area_change")
            overall_an_vs_pd_bb_min_number_of_changes = min(an_vs_pd.get("bb_number_of_changes"), overall_an_vs_pd_bb_min_number_of_changes)
            overall_an_vs_pd_bb_max_number_of_changes = max(an_vs_pd.get("bb_number_of_changes"), overall_an_vs_pd_bb_max_number_of_changes)
            overall_an_vs_pd_bb_sum_number_of_changes += an_vs_pd.get("bb_number_of_changes")
            overall_an_vs_pd_p_min_iou = min(an_vs_pd.get("p_iou"), overall_an_vs_pd_p_min_iou)
            overall_an_vs_pd_p_max_iou = max(an_vs_pd.get("p_iou"), overall_an_vs_pd_p_max_iou)
            overall_an_vs_pd_p_sum_iou += an_vs_pd.get("p_iou")
            overall_an_vs_pd_p_min_percentage_area_change = min(an_vs_pd.get("p_percentage_area_change"), overall_an_vs_pd_p_min_percentage_area_change)
            overall_an_vs_pd_p_max_percentage_area_change = max(an_vs_pd.get("p_percentage_area_change"), overall_an_vs_pd_p_max_percentage_area_change)
            overall_an_vs_pd_p_sum_percentage_area_change += an_vs_pd.get("p_percentage_area_change")
            overall_an_vs_pd_p_min_number_of_changes = min(an_vs_pd.get("p_number_of_changes"), overall_an_vs_pd_p_min_number_of_changes)
            overall_an_vs_pd_p_max_number_of_changes = max(an_vs_pd.get("p_number_of_changes"), overall_an_vs_pd_p_max_number_of_changes)
            overall_an_vs_pd_p_sum_number_of_changes += an_vs_pd.get("p_number_of_changes")

        if count > 0:
            print(f"Statistics for class id: {class_id}")
            print(f"pd_vs_gt bounding box iou, min = {pd_vs_gt_bb_min_iou}, max = {pd_vs_gt_bb_max_iou}, avg = {pd_vs_gt_bb_sum_iou/count}")
            print(f"pd_vs_gt bounding box percentage area change, min = {pd_vs_gt_bb_min_percentage_area_change}, max = {pd_vs_gt_bb_max_percentage_area_change}, avg = {pd_vs_gt_bb_sum_percentage_area_change/count}")
            print(f"pd_vs_gt bounding box number of changes, min = {pd_vs_gt_bb_min_number_of_changes}, max = {pd_vs_gt_bb_max_number_of_changes}, avg = {pd_vs_gt_bb_sum_number_of_changes/count}")
            print(f"pd_vs_gt polygon iou, min = {pd_vs_gt_p_min_iou}, max = {pd_vs_gt_p_max_iou}, avg = {pd_vs_gt_p_sum_iou/count}")
            print(f"pd_vs_gt polygon percentage area change, min = {pd_vs_gt_p_min_percentage_area_change}, max = {pd_vs_gt_p_max_percentage_area_change}, avg = {pd_vs_gt_p_sum_percentage_area_change/count}")
            print(f"pd_vs_gt polygon number of changes, min = {pd_vs_gt_p_min_number_of_changes}, max = {pd_vs_gt_p_max_number_of_changes}, avg = {pd_vs_gt_p_sum_number_of_changes/count}")

            print(f"an_vs_gt bounding box iou, min = {an_vs_gt_bb_min_iou}, max = {an_vs_gt_bb_max_iou}, avg = {an_vs_gt_bb_sum_iou/count}")
            print(f"an_vs_gt bounding box percentage area change, min = {an_vs_gt_bb_min_percentage_area_change}, max = {an_vs_gt_bb_max_percentage_area_change}, avg = {an_vs_gt_bb_sum_percentage_area_change/count}")
            print(f"an_vs_gt bounding box number of changes, min = {an_vs_gt_bb_min_number_of_changes}, max = {an_vs_gt_bb_max_number_of_changes}, avg = {an_vs_gt_bb_sum_number_of_changes/count}")
            print(f"an_vs_gt polygon iou, min = {an_vs_gt_p_min_iou}, max = {an_vs_gt_p_max_iou}, avg = {an_vs_gt_p_sum_iou/count}")
            print(f"an_vs_gt polygon percentage area change, min = {an_vs_gt_p_min_percentage_area_change}, max = {an_vs_gt_p_max_percentage_area_change}, avg = {an_vs_gt_p_sum_percentage_area_change/count}")
            print(f"an_vs_gt polygon number of changes, min = {an_vs_gt_p_min_number_of_changes}, max = {an_vs_gt_p_max_number_of_changes}, avg = {an_vs_gt_p_sum_number_of_changes/count}")

            print(f"an_vs_pd bounding box iou, min = {an_vs_pd_bb_min_iou}, max = {an_vs_pd_bb_max_iou}, avg = {an_vs_pd_bb_sum_iou/count}")
            print(f"an_vs_pd bounding box percentage area change, min = {an_vs_pd_bb_min_percentage_area_change}, max = {an_vs_pd_bb_max_percentage_area_change}, avg = {an_vs_pd_bb_sum_percentage_area_change/count}")
            print(f"an_vs_pd bounding box number of changes, min = {an_vs_pd_bb_min_number_of_changes}, max = {an_vs_pd_bb_max_number_of_changes}, avg = {an_vs_pd_bb_sum_number_of_changes/count}")
            print(f"an_vs_pd polygon iou, min = {an_vs_pd_p_min_iou}, max = {an_vs_pd_p_max_iou}, avg = {an_vs_pd_p_sum_iou/count}")
            print(f"an_vs_pd polygon percentage area change, min = {an_vs_pd_p_min_percentage_area_change}, max = {an_vs_pd_p_max_percentage_area_change}, avg = {an_vs_pd_p_sum_percentage_area_change/count}")
            print(f"an_vs_pd polygon number of changes, min = {an_vs_pd_p_min_number_of_changes}, max = {an_vs_pd_p_max_number_of_changes}, avg = {an_vs_pd_p_sum_number_of_changes/count}")

    if overall_count > 0:
        print(f"Statistics for overall:")
        print(f"overall pd_vs_gt bounding box iou, min = {overall_pd_vs_gt_bb_min_iou}, max = {overall_pd_vs_gt_bb_max_iou}, avg = {overall_pd_vs_gt_bb_sum_iou/overall_count}")
        print(f"overall pd_vs_gt bounding box percentage area change, min = {overall_pd_vs_gt_bb_min_percentage_area_change}, max = {overall_pd_vs_gt_bb_max_percentage_area_change}, avg = {overall_pd_vs_gt_bb_sum_percentage_area_change/overall_count}")
        print(f"overall pd_vs_gt bounding box number of changes, min = {overall_pd_vs_gt_bb_min_number_of_changes}, max = {overall_pd_vs_gt_bb_max_number_of_changes}, avg = {overall_pd_vs_gt_bb_sum_number_of_changes/overall_count}")
        print(f"overall pd_vs_gt polygon iou, min = {overall_pd_vs_gt_p_min_iou}, max = {overall_pd_vs_gt_p_max_iou}, avg = {overall_pd_vs_gt_p_sum_iou/overall_count}")
        print(f"overall pd_vs_gt polygon percentage area change, min = {overall_pd_vs_gt_p_min_percentage_area_change}, max = {overall_pd_vs_gt_p_max_percentage_area_change}, avg = {overall_pd_vs_gt_p_sum_percentage_area_change/overall_count}")
        print(f"overall pd_vs_gt polygon number of changes, min = {overall_pd_vs_gt_p_min_number_of_changes}, max = {overall_pd_vs_gt_p_max_number_of_changes}, avg = {overall_pd_vs_gt_p_sum_number_of_changes/overall_count}")

        print(f"overall an_vs_gt bounding box iou, min = {overall_an_vs_gt_bb_min_iou}, max = {overall_an_vs_gt_bb_max_iou}, avg = {overall_an_vs_gt_bb_sum_iou/overall_count}")
        print(f"overall an_vs_gt bounding box percentage area change, min = {overall_an_vs_gt_bb_min_percentage_area_change}, max = {overall_an_vs_gt_bb_max_percentage_area_change}, avg = {overall_an_vs_gt_bb_sum_percentage_area_change/overall_count}")
        print(f"overall an_vs_gt bounding box number of changes, min = {overall_an_vs_gt_bb_min_number_of_changes}, max = {overall_an_vs_gt_bb_max_number_of_changes}, avg = {overall_an_vs_gt_bb_sum_number_of_changes/overall_count}")
        print(f"overall an_vs_gt polygon iou, min = {overall_an_vs_gt_p_min_iou}, max = {overall_an_vs_gt_p_max_iou}, avg = {overall_an_vs_gt_p_sum_iou/overall_count}")
        print(f"overall an_vs_gt polygon percentage area change, min = {overall_an_vs_gt_p_min_percentage_area_change}, max = {overall_an_vs_gt_p_max_percentage_area_change}, avg = {overall_an_vs_gt_p_sum_percentage_area_change/overall_count}")
        print(f"overall an_vs_gt polygon number of changes, min = {overall_an_vs_gt_p_min_number_of_changes}, max = {overall_an_vs_gt_p_max_number_of_changes}, avg = {overall_an_vs_gt_p_sum_number_of_changes/overall_count}")

        print(f"overall an_vs_pd bounding box iou, min = {overall_an_vs_pd_bb_min_iou}, max = {overall_an_vs_pd_bb_max_iou}, avg = {overall_an_vs_pd_bb_sum_iou/overall_count}")
        print(f"overall an_vs_pd bounding box percentage area change, min = {overall_an_vs_pd_bb_min_percentage_area_change}, max = {overall_an_vs_pd_bb_max_percentage_area_change}, avg = {overall_an_vs_pd_bb_sum_percentage_area_change/overall_count}")
        print(f"overall an_vs_pd bounding box number of changes, min = {overall_an_vs_pd_bb_min_number_of_changes}, max = {overall_an_vs_pd_bb_max_number_of_changes}, avg = {overall_an_vs_pd_bb_sum_number_of_changes/overall_count}")
        print(f"overall an_vs_pd polygon iou, min = {overall_an_vs_pd_p_min_iou}, max = {overall_an_vs_pd_p_max_iou}, avg = {overall_an_vs_pd_p_sum_iou/overall_count}")
        print(f"overall an_vs_pd polygon percentage area change, min = {overall_an_vs_pd_p_min_percentage_area_change}, max = {overall_an_vs_pd_p_max_percentage_area_change}, avg = {overall_an_vs_pd_p_sum_percentage_area_change/overall_count}")
        print(f"overall an_vs_pd polygon number of changes, min = {overall_an_vs_pd_p_min_number_of_changes}, max = {overall_an_vs_pd_p_max_number_of_changes}, avg = {overall_an_vs_pd_p_sum_number_of_changes/overall_count}")

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
    percentage_area_change = abs(ground_truth_polygon.area - predicted_polygon.area)*100 / ground_truth_polygon.area

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
    percentage_area_change = abs(ground_truth_polygon.area - predicted_polygon.area)*100 / ground_truth_polygon.area

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