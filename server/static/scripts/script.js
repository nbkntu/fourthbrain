class AppController {
  constructor(canvasEl, imageEl, submitButtonEl) {
    this.appState = new AppState();
    this.canvasUtil = new CanvasUtil();

    this.canvasEl = canvasEl;
    this.setupCanvasEvents();

    this.submitButtonEl = submitButtonEl;
    this.setupSubmitButtonEvents();

    this.offsetX = 0;
    this.offsetY = 0;
    this.resetOffset();

    this.startX = 0;
    this.startY = 0;

    this.imageEl = imageEl;
  }

  // reset offset coordinates of canvas element
  resetOffset() {
    var box = this.canvasEl.getBoundingClientRect();
    this.offsetX = box.left;
    this.offsetY = box.top;
  }

  setupCanvasEvents() {
    const that = this;
    this.canvasEl.onmousedown = function(e) { that.handleMouseDown(e) };
    this.canvasEl.onmousemove = function(e) { that.handleMouseMove(e) };
    this.canvasEl.onmouseup = function(e) { that.handleMouseUp(e) };
    this.canvasEl.onmouseout = function(e) { that.handleMouseOut(e) };
    this.canvasEl.ondblclick = function(e) { that.handleDoubleClick(e) };
  }

  setupSubmitButtonEvents() {
    const that = this;
    this.submitButtonEl.onclick = function(e) { that.handleSubmitButtonClick(e) };
  }

  handleMouseDown(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

    // record mouse start position
    this.startX = parseInt(e.clientX - this.offsetX);
    this.startY = parseInt(e.clientY - this.offsetY);

    if (this.appState.annotationState == AnnotationState.BOUNDING_BOX) {
      // bounding-box mode
      this.appState.updateRectAndPointClicked(this.startX, this.startY);
      if (this.appState.selectedRectIndex != null && this.appState.rectPointIndex != null) {
        this.appState.isDown = true;
      }
    } if (this.appState.annotationState == AnnotationState.POLYGON) {
      // contour polygon mode
      if (e.button == 0) {  // left click
        this.appState.updatePolygonPointClicked(this.startX, this.startY);
        if (this.appState.polyPointIndex != null){
          this.appState.isDown = true;
        } else {
          this.appState.isDown = false;
        }
      } else if (e.button == 2) {  // right click
        this.appState.updatePolygonPointClicked(this.startX, this.startY);
        // either add or remove points
        if (this.appState.polyPointIndex != null) {
          // right-click on a point -> remove it
          this.appState.removePolyPoint(this.appState.poly, this.appState.polyPointIndex);
          this.appState.trackPolyChanges('remove');
        } else {
          this.appState.addPolyPoint(this.appState.poly, this.startX, this.startY);
          this.appState.trackPolyChanges('add');
        }

        this.draw();
      }
    }
  }

  handleMouseMove(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

    if (!(this.appState && this.appState.isDown)) {
      return;
    }

    const mouseX = parseInt(e.clientX - this.offsetX);
    const mouseY = parseInt(e.clientY - this.offsetY);

    // calculate move distance
    const dx = mouseX - this.startX;
    const dy = mouseY - this.startY;

    // update mouse start position
    this.startX = mouseX;
    this.startY = mouseY;

    // update rect coordinates
    if (this.appState.annotationState == AnnotationState.BOUNDING_BOX) {
      this.appState.updateRectCoordinates(dx, dy);
    } else if (this.appState.annotationState == AnnotationState.POLYGON) {
      this.appState.updatePolyCoordinates(dx, dy);
    }

    this.draw();
  }

  handleMouseUp(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

    if (this.appState.isDown) {
      // update rect coordinates
      if (this.appState.annotationState == AnnotationState.BOUNDING_BOX) {
        this.appState.trackRectChanges();
      } else if (this.appState.annotationState == AnnotationState.POLYGON) {
        this.appState.trackPolyChanges('move');
      }
    }

    // Put your mouseup stuff here
    this.appState.isDown = false;
  }

  handleMouseOut(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

    // Put your mouseOut stuff here
    this.appState.isDown = false;
  }

  handleDoubleClick(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

    const mouseX = parseInt(e.clientX - this.offsetX);
    const mouseY = parseInt(e.clientY - this.offsetY);

    // double click inside bounding box rectangle
    if (this.appState.annotationState == AnnotationState.BOUNDING_BOX) {
      var rectIds = this.appState.getRectContains(mouseX, mouseY);
      if (!rectIds || !rectIds.length) {
        return;
      }
      if (rectIds.length > 1) {
        console.log('more than one bounding boxes contain the point');
        return;
      }

      const rid = rectIds[0];
      const rect = this.appState.rects[rid];
      const objectClass = this.appState.objectClasses[rid];
      // save selected bounding box
      this.appState.selectedRectIndex = rid;
      // switch to polygon mode
      this.appState.annotationState = AnnotationState.POLYGON;
      // get object boundary
      this.getObjectBoundary(this.appState.filename, rect, objectClass);
    }

    // Put your mouseup stuff here
    this.appState.isDown = false;
  }

  handleSubmitButtonClick(e) {
    const result = this.appState.getAnnotationResult();
    console.log(result);

    this.submitAnnotationResult(this.appState.filename, result);
  }

  draw() {
    const ctx = this.getCanvasContext();

    // clear canvas
    this.canvasUtil.clearRect(ctx, 0, 0, canvas.width, canvas.height);

    // draw image html element onto canvas, keep image original size
    this.canvasUtil.drawImage(ctx, this.imageEl, 0, 0);

    if (this.appState.annotationState == AnnotationState.BOUNDING_BOX) {
      this.canvasUtil.drawBoundingBoxes(ctx, this.appState.rects, this.appState.dotSize);
    } else if (this.appState.annotationState == AnnotationState.POLYGON) {
      this.canvasUtil.drawContour(ctx, this.appState.poly, this.appState.dotSize);
    }
  }

  getCanvasContext() {
    const canvas = this.canvasEl;
    return canvas.getContext('2d');
  }

  handleImageLoaded(fileObj, toUploadImage) {
    // then get bounding box
    const that = this;
    if (toUploadImage) {
      // first upload image
      uploadImage(fileObj).then(
          function(resp) {
            console.log(resp);
            const filename = resp.filename;
            console.log('uploaded:', filename);

            // set the new filename
            that.appState.filename = filename;

            getBoundingBoxes(filename).then(
                function(resp) {
                  that.getBoundingBoxesCallback(resp);
                },
                function(error) {
                    console.log("Error: ", error);
                });
          },
          function(error) {
              console.log("Error: ", error);
          });
    } else {
      filename = fileObj.name;
      console.log(filename);
      this.appState.filename = filename;

      getBoundingBoxes(filename).then(
          function(resp) {
              that.getBoundingBoxesCallback(resp);
          },
          function(error) {
              console.log("Error: ", error);
          });
    }
  }

  getBoundingBoxesCallback(resp) {
    console.log(resp);

    if (resp.bounding_box && resp.bounding_box.length) {
      var rects = [];
      resp.bounding_box.forEach((bb) => {
        const rect = {
          x1: bb[0],
          y1: bb[1],
          x2: bb[2],
          y2: bb[3]
        }
        rects.push(rect);
      });
      console.log(rects);

      this.appState.rects = rects;
      this.appState.objectClasses = resp.classes;
      // deep copy
      this.appState.orgRects= JSON.parse(JSON.stringify(this.appState.rects));
    } else {
      console.log('no bounding boxes');
    }

    this.appState.annotationState = AnnotationState.BOUNDING_BOX;

    this.draw();
  }

  getObjectBoundary(filename, boundingBox, objectClass) {
    console.log(filename);

    var that = this;
    getObjectBoundaries(filename, boundingBox, objectClass).then(
      function(resp) {
          that.getObjectBoundaryCallback(resp);
      },
      function(error) {
          console.log("Error: ", error);
      });
  }

  getObjectBoundaryCallback(resp) {
    console.log(resp);

    var points = [];

    resp.simple_mask_polygon.forEach((p) => {
      var point = {
        x: p[0],
        y: p[1]
      }
      points.push(point);
    });

    console.log(points);
    this.appState.poly.points = points;
    // deep copy
    this.appState.orgPoly= JSON.parse(JSON.stringify(this.appState.poly));

    this.draw();
  }

  submitAnnotationResult(filename, result) {
    console.log(filename);

    var that = this;
    submitResult(filename, result).then(
      function(resp) {
        that.submitAnnotationResultCallback(resp);
      },
      function(error) {
          console.log("Error: ", error);
      });
  }

  submitAnnotationResultCallback(resp) {
    console.log(resp);
    this.appState.AnnotationState = AnnotationState.DONE;
  }

};

class CanvasUtil {
  constructor() {}

  clearRect(ctx, x, y, width, height) {
    ctx.clearRect(x, y, width, height);
  }

  drawImage(ctx, img, x, y) {
    ctx.drawImage(img, 0, 0);
  }

  drawPointRect(ctx, x, y, dotSize) {
    ctx.rect(
        x - dotSize / 2,
        y - dotSize / 2,
        dotSize,
        dotSize
    );
    ctx.stroke();
  }

  drawBoundingBox(ctx, rect, dotSize) {
    // draw bounding box rectangle
    ctx.beginPath();
    ctx.rect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
    ctx.strokeStyle = 'yellow';
    ctx.stroke();

    // draw corner boxes
    this.drawPointRect(ctx, rect.x1, rect.y1, dotSize);
    this.drawPointRect(ctx, rect.x1, rect.y2, dotSize);
    this.drawPointRect(ctx, rect.x2, rect.y1, dotSize);
    this.drawPointRect(ctx, rect.x2, rect.y2, dotSize);
  }

  drawBoundingBoxes(ctx, rects, dotSize) {
    if (rects && rects.length) {
      rects.forEach((rect) => {
        this.drawBoundingBox(ctx, rect, dotSize);
      });
    }
  }

  drawContour(ctx, poly, dotSize) {
    ctx.beginPath();

    // draw polygon
    ctx.beginPath();  // need this for clearRect to work

    ctx.moveTo(poly.points[0].x, poly.points[0].y);
    for (var i = 0; i < poly.points.length; i++) {
      ctx.lineTo(poly.points[i].x, poly.points[i].y);
    }
    ctx.lineTo(poly.points[0].x, poly.points[0].y);

    ctx.strokeStyle = 'yellow';
    ctx.stroke();

    // draw corner boxes
    for (var i = 0; i < poly.points.length; i++) {
      // draw rectangle at the points
      this.drawPointRect(ctx, poly.points[i].x, poly.points[i].y, dotSize);
    }
  }
};

class AppState {
  constructor() {
    this.isDown = false;

    this.filename = '';

    // selected bounding box rectangle
    this.selectedRectIndex = null;

    // currently seelected point index of bounding box rectangle
    this.rectPointIndex = null;

    this.annotationState = AnnotationState.START;

    // size of control rectangle
    this.dotSize = 8;

    this.objectClasses = [];

    // original bounding boxes
    this.orgRects = [];

    // bounding box rectangles
    this.rects = [{
      x1: 20,
      y1: 20,
      x2: 300,
      y2: 200
    }];

    // map object that track number of changes for bounding boxes
    this.rectChanges = {};

    // original contour polygon
    this.orgPoly = {};

    // contour polygon
    this.poly = {
      points:[
        {x: 144, y: 147},
        {x: 180, y: 128},
        {x: 205, y: 110},
        {x: 216, y: 123},
        {x: 310, y: 54},
        {x: 406, y: 45},
        {x: 509, y: 45},
        {x: 594, y: 72},
        {x: 658, y: 119},
        {x: 652, y: 209},
        {x: 643, y: 308},
        {x: 597, y: 295},
        {x: 577, y: 306},
        {x: 546, y: 310},
        {x: 487, y: 328},
        {x: 418, y: 351},
        {x: 369, y: 337},
        {x: 252, y: 330},
        {x: 167, y: 319},
        {x: 145, y: 317},
        {x: 122, y: 338},
        {x: 93, y: 333},
        {x: 71, y: 307},
        {x: 62, y: 267},
        {x: 67, y: 214},
        {x: 97, y: 171}
      ],
    }

    // track poly changes
    this.polyChanges = {
      move: 0,
      add: 0,
      delete: 0
    };
  }

  updateRectCoordinates(dx, dy) {
    if (this.selectedRectIndex == null || this.rectPointIndex == null) {
      return;
    }
    var rect = this.rects[this.selectedRectIndex];
    switch(this.rectPointIndex) {
      case 1:
        rect.x1 += dx;
        rect.y1 += dy;
        break;
      case 2:
        rect.x2 += dx;
        rect.y1 += dy;
        break;
      case 3:
        rect.x1 += dx;
        rect.y2 += dy;
        break;
      case 4:
        rect.x2 += dx;
        rect.y2 += dy;
        break;
    }
  };

  trackRectChanges() {
    if (this.selectedRectIndex == null || this.rectPointIndex == null) {
      return;
    }
    // track changes
    if (this.selectedRectIndex in this.rectChanges) {
      this.rectChanges[this.selectedRectIndex] += 1;
    } else {
      this.rectChanges[this.selectedRectIndex] = 1;
    }
  }

  updatePolyCoordinates(dx, dy) {
    if (this.polyPointIndex == null) {
      return;
    }
    this.poly.points[this.polyPointIndex].x += dx;
    this.poly.points[this.polyPointIndex].y += dy;
  };

  removePolyPoint(poly, pointIndex) {
    if (poly.points.length <= 3) {
      return;
    }

    poly.points.splice(pointIndex, 1);
  }

  addPolyPoint(poly, x, y) {
    var insertIndex = -1;
    var minSquareDistance = -1;
    for (var i = 0; i < poly.points.length; i++) {
      var i1 = (i + 1) % poly.points.length;
      var sqDist = Math.pow(x - poly.points[i].x, 2) + Math.pow(y - poly.points[i].y, 2)
          + Math.pow(x - poly.points[i1].x, 2) + Math.pow(y - poly.points[i1].y, 2);
      if (minSquareDistance == -1 || minSquareDistance > sqDist) {
        minSquareDistance = sqDist;
        insertIndex = i1;
      }
    }

    poly.points.splice(insertIndex, 0, {x: x, y: y});
  }

  trackPolyChanges(changeType) {
    switch (changeType) {
      case 'move':
        this.polyChanges.move += 1;
        return;
      case 'add':
        this.polyChanges.add += 1;
        return;
      case 'remove':
        this.polyChanges.delete += 1;
        return;
    }
  }

  inClickRange(x0, y0, dotSize, x, y) {
    const x1 = x0 - dotSize/2;
    const x2 = x0 + dotSize/2;
    const y1 = y0 - dotSize/2;
    const y2 = y0 + dotSize/2;
    if (x1 <= x && x <= x2 && y1 <= y && y <= y2) {
      return true;
    }
    return false;
  };

  getPointClicked(rect, dotSize, x, y) {
    // 1 2
    // 3 4
    if (this.inClickRange(rect.x1, rect.y1, dotSize, x, y)) {
      return 1;
    }
    if (this.inClickRange(rect.x2, rect.y1, dotSize, x, y)) {
      return 2;
    }
    if (this.inClickRange(rect.x1, rect.y2, dotSize, x, y)) {
      return 3;
    }
    if (this.inClickRange(rect.x2, rect.y2, dotSize, x, y)) {
      return 4;
    }
    return null;
  };

  updateRectAndPointClicked(x, y) {
    for (var i = 0; i < this.rects.length; i++) {
      const p = this.getPointClicked(this.rects[i], this.dotSize, x, y);
      if (p != null) {
        this.selectedRectIndex = i;
        this.rectPointIndex = p;
        return;
      }
    }
    this.selectedRectIndex = null;
    this.rectPointIndex = null;
  }

  updatePolygonPointClicked(x, y) {
    const poly = this.poly;
    const dotSize = this.dotSize;

    for (var i = 0; i < poly.points.length; i++) {
      const x1 = poly.points[i].x - dotSize/2;
      const x2 = poly.points[i].x + dotSize/2;
      const y1 = poly.points[i].y - dotSize/2;
      const y2 = poly.points[i].y + dotSize/2;
      if (x1 <= x && x2 >= x && y1 <= y && y2 >= y) {
        this.polyPointIndex = i;
        return;
      }
    }
    this.polyPointIndex = null;
  }

  getRectContains(x, y) {
    var rectIds = [];
    this.rects.forEach((rect, i) => {
      if (rect.x1 <= x && rect.x2 >=x && rect.y1 <= y && rect.y2 >= y) {
        rectIds.push(i);
      }
    });
    return rectIds;
  }

  getAnnotationResult() {
    const objectClass = this.objectClasses[this.selectedRectIndex];

    const rp = this.orgRects[this.selectedRectIndex];
    const predictedBoundingBox = [rp.x1, rp.y1, rp.x2, rp.y2];

    const predictedPolygon = [];
    this.orgPoly.points.forEach((p) => {
      predictedPolygon.push([p.x, p.y]);
    });

    const ra = this.rects[this.selectedRectIndex];
    const annotatedBoundingBox = [ra.x1, ra.y1, ra.x2, ra.y2];

    const annotatedPolygon = [];
    this.poly.points.forEach((p) => {
      annotatedPolygon.push([p.x, p.y]);
    });

    const boundingBoxChanges = this.rectChanges && this.rectChanges[this.selectedRectIndex]
        ? this.rectChanges[this.selectedRectIndex] : 0;

    const polyChanges = this.polyChanges.move + this.polyChanges.add + this.polyChanges.delete;

    return {
      object_class: objectClass,
      predicted_bounding_box: predictedBoundingBox,
      predicted_polygon: predictedPolygon,
      annotated_bounding_box: annotatedBoundingBox,
      annotated_polygon: annotatedPolygon,
      bounding_box_changes: boundingBoxChanges,
      polygon_changes: polyChanges
    };
  }
};

class AnnotationState {
  static START = 'start';
  static BOUNDING_BOX = 'bounding-box';
  static POLYGON = 'polygon';
  static DONE = 'done';
}

const uploadImage = async (fileObj) => {
  console.log('uploadImage');
  console.log(fileObj);

  var formData = new FormData();
  formData.append("file", fileObj);

  const response = await fetch('/upload_image', {
    method: 'POST',
    body: formData,
  });

  let resp = await response.json();
  return resp;
};

const getBoundingBoxes = async (filename) => {
    const response = await fetch('/get_bounding_boxes', {
      method: 'POST',
      body: `{
        "image_id": "img1",
        "image_file_name": "${filename}"
      }`,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // extract JSON from the http response
    const resp = await response.json();
    return resp;
};

const getObjectBoundaries = async (filename, boundingBox, objectClass) => {
  const response = await fetch('/get_object_boundary', {
    method: 'POST',
    body: `{
      "image_id": "img1",
      "image_file_name": "${filename}",
      "bounding_box": [${boundingBox.x1}, ${boundingBox.y1}, ${boundingBox.x2}, ${boundingBox.y2}],
      "class_of_interest": ${objectClass}
    }`,
    headers: {
      'Content-Type': 'application/json'
    }
  });

  // extract JSON from the http response
  const resp = await response.json();
  return resp;
};

const submitResult = async (filename, result) => {
  const req = {
    image_id: 'img1',
    image_file_name: filename,
    result: result
  }
  const response = await fetch('/submit_result', {
    method: 'POST',
    body: JSON.stringify(req),
    headers: {
      'Content-Type': 'application/json'
    }
  });

  // extract JSON from the http response
  const resp = await response.json();
  return resp;
};
