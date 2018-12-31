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


Viewport.prototype.clear = function(color)
{
    if (color != undefined)
        this.context.fillStyle = color;
    this.context.fillRect(0, 0, this.width, this.height);
}

Viewport.prototype.setFont = function(font)
{
    this.context.font = font;
}


Viewport.prototype.print = function(text, x, y, color)
{
    if (color != undefined)
        this.context.fillStyle = color;
    this.context.fillText(text, x, y);
}



// Maps a point in 2D space to the viewport.
// Returns a Vector2d.
Viewport.prototype.mapToViewport = function(vector, gridScale=1)
{
    // Invert Y value b/c positive Y is down in the viewport.
    var result = [];
    result[0] = ( vector.x * gridScale) + this._centerX;
    result[1] = (-vector.y * gridScale) + this._centerY;
    result[2] = 0;
    return new Vector(result);
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
Viewport.prototype.drawGrid = function(axisColor)
{
    this.context.lineWidth = 1;
    this.context.strokeStyle = axisColor;

    // X axis
    this.drawLine(0, this._centerY, this.width, this._centerY);

    // Y axis
    this.drawLine(this._centerX, 0, this._centerX, this.height);
}