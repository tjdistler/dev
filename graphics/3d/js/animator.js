function Animator(viewport, scene)
{
    this.scene = new Scene(scene);
    this.viewport = viewport;
    
    this.viewport.setFont('12px Courier New');

    this.loop = false;
    this.hud = false;
    this.active = false;
    this.sceneTime = 0; //ms
    this.frameStep = 0 //ms
    this.startWallTime = 0;
    this.prevWallTime = 0;
    this.sleepTime = 0;
}


Animator.prototype.play = function()
{
    this.active = true;
    this.sceneTime = 0; //ms
    this.frameStep = 1 / this.scene.fps * 1000; //ms
    this.startWallTime = Date.now();
    this.prevWallTime = this.startWallTime;
    this.sleepTime = this.frameStep;

    this.renderNextFrame(0); // Render first frame

    // Setup timer to render next frame.
    setTimeout(this.timerCb.bind(this), this.sleepTime);
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
    
    // Always stop on exactly the last frame
    this.sceneTime = Math.min(now - this.startWallTime, this.scene.duration);

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

// Renders the scene to the viewport for the given scene time.
Animator.prototype.renderNextFrame = function(sceneTime)
{
    var vertexCount = this.scene.render(sceneTime, this.viewport);
    
    // Draw Heads-up-Display
    if (this.hud)
    {
        this.viewport.drawGrid('#fff');
        this.viewport.print('t:' + sceneTime.toString(), 5, 14, 'yellow');
        this.viewport.print('v:' + vertexCount.toString(), 5, 27, 'yellow');
    }
}