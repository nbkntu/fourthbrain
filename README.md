# fourthbrain

## server

- Install dependencies with conda
```
$ cd server/yolov3_tf2
$ conda env create -f conda-cpu.yml
```

- Launch server
```
$ cd server
$ uvicorn main:app --reload
```
