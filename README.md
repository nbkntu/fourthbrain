# fourthbrain capstone project

## smart annotation server

### Local
#### 1. YOLOv3 setup
- Install dependencies with conda (conda environment "smartannotation" will be created)
```
$ cd server
$ conda env create -f conda-cpu.yml
```

- Download YOLOv3 pre-trained model
https://pjreddie.com/media/files/yolov3.weights
and put in under `server/yolov3_tf2/model`


#### 2. Mask RCNN setup
- Download Mask RCNN pre-trained model
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
and put in under `server/mask_rcnn/model`

#### 3. Launch server locally
```
$ cd server
$ conda activate smartannotation
$ uvicorn main:app --reload
```

### Deploy to Elastic Beanstalk

Make sure Elastic Beanstalk CLI has already been installed.

```
$ cd server
```

- create CPU instance
```
$ eb create smart-annotation-test --instance_type=t2.large
```

- create GPU instance
```
$ eb create smart-annotation-gpu-test --instance_type=p2.xlarge

```

- redeploy the server content
```
$ eb deploy smart-annotation-test
```

- get instance logs
```
$ eb logs smart-annotation-test
```

- terminate instance
```
$ eb terminiate smart-annotation-test
```
