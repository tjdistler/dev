function Animator(viewport, scene)
{
  this.SHOW_GRID = true;
  this.GRID_COLOR = "#eeeeee";
  this.GRID_AXIS_COLOR = "#bbbbbb";
  this.GRID_SCALE = 25; //pixels per division

  this.scene = new Scene(scene);
  this.viewport = viewport;

  this.active = false;
  this.sceneTime = 0; //ms
  this.frameStep = 1 / this.scene.fps * 1000; //ms
  this.startWallTime = Date.now();
  this.prevWallTime = this.startWallTime;
  this.sleepTime = this.frameStep;
}


Animator.prototype.play = function(loop = false)
{
  this.loop = loop;
  this.active = true;

  this.renderNextFrame(0); // Render first frame

  // Setup timer to render next frame.
  setTimeout(this.timerCb.bind(this), this.sleepTime); // compensate for render time.
}


Animator.prototype.stop = function()
{
    this.loop = false;
    this.active = false;
}

// Handles scene time updates and resetting the callback timer.
Animator.prototype.timerCb = function()
{
  if (!this.active)
    return;

  var now = Date.now();
  this.sceneTime = now - this.startWallTime;

  this.renderNextFrame(this.sceneTime);

  // Stop setting timers when complete, or loop
  if (this.sceneTime >= this.scene.duration)
  {
    if (this.active && this.loop)
    {
      this.sceneTime = 0;
      this.startWallTime = this.prevWallTime = Date.now();
      this.sleepTime = this.frameStep;
      setTimeout(this.timerCb.bind(this), this.sleepTime);
    }

    return;
  }

  // Reset the timer. Compensate for render time and timer jitter.
  this.sleepTime += this.frameStep;
  this.sleepTime -= now - this.prevWallTime;
  this.prevWallTime = now;
  setTimeout(this.timerCb.bind(this), this.sleepTime);
}

// Rasterizes the scene onto the frame buffer.
Animator.prototype.renderNextFrame = function(sceneTime)
{
  this.viewport.clear();

  if (this.SHOW_GRID)
    this.viewport.drawGrid(this.GRID_AXIS_COLOR, this.GRID_COLOR, this.GRID_SCALE);

  var geometry = this.scene.generateFrameGeometry(sceneTime);
    
  // TODO: clip
    
  // Draw each triangle
  geometry.forEach(function(triangle) {
    var a = this.viewport.mapToViewport(new Vector2d(triangle[0]), this.GRID_SCALE);
    var b = this.viewport.mapToViewport(new Vector2d(triangle[1]), this.GRID_SCALE);
    var c = this.viewport.mapToViewport(new Vector2d(triangle[2]), this.GRID_SCALE);

    this.viewport.drawLine(a.x, a.y, b.x, b.y, '#000', 1); 
    this.viewport.drawLine(b.x, b.y, c.x, c.y, '#000', 1);
    this.viewport.drawLine(c.x, c.y, a.x, a.y, '#000', 1);
  }, this);

  //viewport.render();
}