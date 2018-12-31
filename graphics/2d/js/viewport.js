
// Represent a canvas.
// elementId - The HTML element 'id' of the canvas to use.
function Viewport(elementId)
{
  this.canvas  = document.getElementById(elementId);
  this.context = this.canvas.getContext('2d');

  this.width = this.canvas.width;
  this.height = this.canvas.height;
  this.frameBuffer = this.context.createImageData(this.width, this.height);

  // Note: Must add 0.5 to prevent 1 pixel lines from spanning 2 pixels due to the 
  // center point of the line being exactly on the division between 2 pixels.
  this._centerX = (this.width  / 2) + 0.5;
  this._centerY = (this.height / 2) + 0.5;
}


Viewport.prototype.render = function()
{
  this.context.putImageData(this.frameBuffer, 0, 0);
}


Viewport.prototype.clear = function()
{
  this.context.clearRect(0, 0, this.width, this.height);
}


// Maps a point in 2D space to the viewport.
// Returns a Vector2d.
Viewport.prototype.mapToViewport = function(vector, gridScale=1)
{
  // Invert Y value b/c positive Y is down in the viewport.
  var result = [];
  result[0] = ( vector.x * gridScale) + this._centerX;
  result[1] = (-vector.y * gridScale) + this._centerY;
  return new Vector2d(result);
}


// Draw a line from (x1,y1) t0 (x2,y2). Color and width are optional.
Viewport.prototype.drawLine = function(x1, y1, x2, y2, color, width)
{
  if (width !== undefined)
    this.context.lineWidth = width;

  if (color !== undefined)
    this.context.strokeStyle = color;

  this.context.beginPath();
  this.context.moveTo(x1, y1);
  this.context.lineTo(x2, y2);
  this.context.stroke();
}


// Draws an x/y grid over the rendered frame for reference.
// axisColor - String #rrggbb defining the color of the axis lines.
// gridColor - String #rrggbb defining the color of the grid lines.
// gridScale - The number of pixels between grid lines.
Viewport.prototype.drawGrid = function(axisColor, gridColor, gridScale)
{
  this.context.lineWidth = 1;
  this.context.strokeStyle = axisColor;

  // X axis
  this.drawLine(0, this._centerY, this.width, this._centerY);

  // Y axis
  this.drawLine(this._centerX, 0, this._centerX, this.height);

  // Draw grid lines.
  // color - String '#rrggbb' that defines the grid line colore.
  // generator(idx) - Function to return the next line location based on the input 'idx'.
  //     Should return undefined when no more lines should be generated. 
  //     The returned value must be in the format {x1,y1,x2,y2}
  var drawGridLine = function(color, generator)
  {
    for (ii=1; true; ++ii)
    {
      var line = generator(ii);
      if (line === undefined)
          break;
      this.drawLine(line.x1, line.y1, line.x2, line.y2, color);
    }
  }.bind(this);

  // positive horizontal lines
  drawGridLine(gridColor, function(idx) {
    var y = (this._centerY)-(idx*gridScale);
    return (y >= 0) ? {x1:0, y1:y, x2:this.width, y2:y} : undefined;
  }.bind(this));

  // negative horizontal lines
  drawGridLine(gridColor, function(idx) {
    var y = (this._centerY)+(idx*gridScale);
    return (y <= this.height) ? {x1:0, y1:y, x2:this.width, y2:y} : undefined;
  }.bind(this));

  // positive vertical lines
  drawGridLine(gridColor, function(idx) {
    var x = (this._centerX)+(idx*gridScale);
    return (x <= this.width) ? {x1:x, y1:0, x2:x, y2:this.height} : undefined;
  }.bind(this));

  // negative vertical lines
  drawGridLine(gridColor, function(idx) {
    var x = (this._centerX)-(idx*gridScale);
    return (x >= 0) ? {x1:x, y1:0, x2:x, y2:this.height} : undefined;
  }.bind(this));
}
