# fourthbrain

## server

### YOLOv3 setup
- Install dependencies with conda
```
$ cd server
$ conda env create -f conda-cpu.yml
```

- Download YOLOv3 pre-trained model
https://pjreddie.com/media/files/yolov3.weights
and put in under `server/yolov3_tf2/data`

- Then run convert.py
```
$ cd server/yolov3_tf
$ python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

### Mask RCNN setup
- Download Mask RCNN pre-trained model
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
and put in under `server/mask_rcnn/model`

- Launch server
```
$ cd server
$ uvicorn main:app --reload
```
