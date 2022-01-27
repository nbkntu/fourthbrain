class AppController {
  constructor(canvasEl, imageEl) {
    this.appState = new AppState();
    this.canvasUtil = new CanvasUtil();

    this.canvasEl = canvasEl;
    this.setupCanvasEvents();

    this.offsetX = 0;
    this.offsetY = 0;
    this.resetOffset();

    this.startX = 0;
    this.startY = 0;

    this.imageEl = imageEl;
  }

  // reset offset coordinates of canvas element
  resetOffset() {
    var box = canvas.getBoundingClientRect();
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

  handleMouseDown(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

    // record mouse start position
    this.startX = parseInt(e.clientX - this.offsetX);
    this.startY = parseInt(e.clientY - this.offsetY);

    if (this.appState.annotationMode == 'bounding-box') {
      // bounding-box mode
      this.appState.rectPointIndex = this.canvasUtil.getPointClicked(
          this.appState.rect, this.appState.dotSize, this.startX, this.startY);
      if (this.appState.rectPointIndex != null) {
        this.appState.isDown = true;
      }
    } if (this.appState.annotationMode == 'contour') {
      // contour polygon mode
      if (e.button == 0) {  // left click
        this.appState.polyPointIndex = this.canvasUtil.getPointClickedPolygon(
            this.appState.poly, this.appState.dotSize, this.startX, this.startY);
        if (this.appState.polyPointIndex != null){
          this.appState.isDown=true;
        }
      } else if (e.button == 2) {  // right click
        // either add or remove points
        this.appState.polyPointIndex = this.canvasUtil.getPointClickedPolygon(
            this.appState.poly, this.appState.dotSize, this.startX, this.startY);
        if (this.appState.polyPointIndex != null) {
          // right-click on a point -> remove it
          this.appState.removePolyPoint(this.appState.poly, this.appState.polyPointIndex);
        } else {
          this.appState.addPolyPoint(this.appState.poly, this.startX, this.startY);
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
    if (this.appState.annotationMode == 'bounding-box') {
      this.appState.updateRectCoordinates(dx, dy);
    } else if (this.appState.annotationMode == 'contour') {
      this.appState.updatePolyCoordinates(dx, dy);
    }

    this.draw();
  }

  handleMouseUp(e) {
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();

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
    const rect = this.appState.rect;
    if (this.appState.annotationMode == 'bounding-box') {
      if (mouseX >= rect.x1 && mouseX <= rect.x2
        && mouseY >= rect.y1 && mouseY <= rect.y2) {
          // switch to polygon mode
          this.appState.annotationMode = 'contour';
          this.draw();
      }
    }

    // Put your mouseup stuff here
    this.appState.isDown = false;
  }

  draw() {
    const ctx = this.getCanvasContext();

    // clear canvas
    this.canvasUtil.clearRect(ctx, 0, 0, canvas.width, canvas.height);

    // draw image html element onto canvas, keep image original size
    this.canvasUtil.drawImage(ctx, this.imageEl, 0, 0);

    if (this.appState.annotationMode == 'bounding-box') {
      this.canvasUtil.drawBoundingBox(ctx, this.appState.rect, this.appState.dotSize);
    } else if (this.appState.annotationMode == 'contour') {
      this.canvasUtil.drawContour(ctx, this.appState.poly, this.appState.dotSize);
    }
  }

  getCanvasContext() {
    const canvas = this.canvasEl;
    return canvas.getContext('2d');
  }

  getBoundingBox(ctx, filename) {
    console.log(filename);
    var that = this;
    getBoundingBoxes(filename).then(
      function(resp) {
          that.getBoundingBoxCallback(resp);
      },
      function(error) {
          console.log("Error: ", error);
      });
  }

  getBoundingBoxCallback(resp) {
    console.log(resp);

    var rect = this.appState.rect;
    rect.x1 = resp.bounding_box[0][0];
    rect.y1 = resp.bounding_box[0][1];
    rect.x2 = resp.bounding_box[0][2];
    rect.y2 = resp.bounding_box[0][3];

    console.log(rect);

    this.appState.annotationMode = 'bounding-box';

    this.draw();
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

  drawContour(ctx, poly, dotSize) {
    ctx.beginPath();

    // draw polygon
    ctx.beginPath();  // need this for clearRect to work

    ctx.moveTo(poly.points[0].x,poly.points[0].y);
    for (var i = 0; i < poly.points.length; i++) {
      ctx.lineTo(poly.points[i].x,poly.points[i].y);
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

  getPointClickedPolygon(poly, dotSize, x, y) {
    for (var i = 0; i < poly.points.length; i++) {
      const x1 = poly.points[i].x - dotSize/2;
      const x2 = poly.points[i].x + dotSize/2;
      const y1 = poly.points[i].y - dotSize/2;
      const y2 = poly.points[i].y + dotSize/2;
      if (x1 <= x && x2 >= x && y1 <= y && y2 >= y) {
        return i;
      }
    }
    return null;
  }
};

class AppState {
  constructor() {
    this.isDown = false;

    // currently seelected point index of bounding box rectangle
    this.rectPointIndex = null;

    // current mode: 'ready' / 'bounding-box' / 'polygon'
    this.annotationMode = 'ready';

    // size of control rectangle
    this.dotSize = 8;

    // bounding box rectangle
    this.rect = {
      x1: 20,
      y1: 20,
      x2: 300,
      y2: 200
    };

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
  }

  updateRectCoordinates(dx, dy) {
    if (!this.rectPointIndex) {
      return;
    }
    var rect = this.rect;
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
