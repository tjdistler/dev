// Represents a scene to render. A scene contains all the information needed
// to render and animate the ojects in the scene.
//
// data - The scene object to render
function Scene(scene)
{
    this.scene = scene;
    this.fps = this.scene.fps;
    this.duration = this.scene.duration;
    
    this.renderer = new Renderer();
    this.renderer.setShaders(VertShaderGouraud, {}, FragShaderGouraud, {});
}


/* Renders the scene to the specified buffer using the render API.
 *
 * sceneTime - Time (in milliseconds) in the scene to render.
 * buffer - The ImageData buffer to render the scene into. The pixel format
 *      is assumed to be 4-byte RGBA data.
 *
 * Returns the number of verticies rendered.
 */
Scene.prototype.render = function(sceneTime, viewport)
{
    // Setup the render state.
    this.renderer.setCamera( this._getCamera(sceneTime) );
    this.renderer.setLights( this._getLights(sceneTime) );
    
    var vertices = this._getWorldSpaceVertices(sceneTime);
    this.renderer.setVertices(vertices);

    this.renderer.execute({rasterize:false}, []);
    
    this._renderWireframe(vertices, viewport);
        
    return vertices.length * 3;
}


/* 
 * Returns a Camera object representing the camera state at the specified
 * scene time.
 */
Scene.prototype._getCamera = function(sceneTime)
{
    //TODO - handle camera transforms
    
    return new Camera(new Vector(this.scene.camera.position),
                      new Vector(this.scene.camera.look_at),
                      this.scene.camera.fov,
                      this.scene.width/this.scene.height);
}


/* 
 * Returns a array of Light objects representing the lights in the scene at
 * the specified scene time.
 */
Scene.prototype._getLights = function(sceneTime)
{
    //TODO
}


/* Transforms the scene geometry into a list of triangles in world space.
 *
 * sceneTime - Time (in milliseconds) in the scene to render.
 *
 * Returns an array of triangle Vertex objects of the form:
 *  [ [[Vertex,Vertex,Vertex], [Vertex,Vertex,Vertex], [Vertex,Vertex,Vertex]], ...]
 */
Scene.prototype._getWorldSpaceVertices = function(sceneTime)
{
    //TODO
    return this.generateFrameGeometry(sceneTime);
}


// Returns an array of triangles (in world space) to be rendered. This function
// performs all vertex transformations into world space.
// sceneTime - ms since the beginning of the scene.
// Return value in the form [ [[x,y],[x,y],[x,y]], [...], ... ]
Scene.prototype.generateFrameGeometry = function(sceneTime)
{
    var output = [];

    // Process each object
    this.scene.objects.forEach(function(obj, objIdx) {

        // Check to see if the object exists during this sceneTime.
        // Undefined begin/end times mean 'always exists'.
        if (obj.begin_ts !== undefined && sceneTime < obj.begin_ts)
            return;
        if (obj.end_ts !== undefined && sceneTime > obj.end_ts)
            return;

        // All transformations happen on a copy of the object geometry.
        var triangles = deepCopy(obj.triangles);

        // Process each transform in the object.
        obj.transforms.forEach(function(transform) {

            // Is it time to apply the transform yet?
            if (sceneTime <= transform.begin_ts)
                return;

            // Calculate what percentage of the transform applies at the current sceneTime.
            // Note: undefined begin/end times mean the transforms are to be applied over the
            // entire scene duration.
            var transBegin = transform.begin_ts !== undefined ? transform.begin_ts : 0;
            var transEnd = transform.end_ts !== undefined ? transform.end_ts : this.scene.duration;

            var transformTime = sceneTime - transBegin;
            var transformLen = transEnd - transBegin;
            var transformPercentage = Math.min(transformTime / transformLen, 1.0);
    
            if (transform.type == 'rotate')
            {
                var dx = transform.dx !== undefined ? transform.dx * transformPercentage : 0;
                var dy = transform.dy !== undefined ? transform.dy * transformPercentage : 0;
                var dz = transform.dz !== undefined ? transform.dz * transformPercentage : 0;
                triangles.forEach(function(triangle, idx) {
                    triangles[idx] = this.rotate(dx, dy, dz, triangle);
                }, this);
            }
            else if (transform.type == 'scale')
            {
                // Note: scaling starts at 1, not 0
                var magnitudeX = transform.dx ? transform.dx - 1 : 0;
                var magnitudeY = transform.dy ? transform.dy - 1 : 0;
                var magnitudeZ = transform.dz ? transform.dz - 1 : 0;
                var dx = 1 + (magnitudeX * transformPercentage);
                var dy = 1 + (magnitudeY * transformPercentage);
                var dz = 1 + (magnitudeZ * transformPercentage);

                triangles.forEach(function(triangle, idx) {
                    triangles[idx] = this.scale(dx, dy, dz, triangle);
                }, this);
            }
            else if (transform.type == 'translate')
            {
                var dx = transform.dx ? transform.dx * transformPercentage : 0;
                var dy = transform.dy ? transform.dy * transformPercentage : 0;
                var dz = transform.dz ? transform.dz * transformPercentage : 0;

                triangles.forEach(function(triangle, idx) {
                    triangles[idx] = this.translate(dx, dy, dz, triangle);
                }, this);
            }
            else
            {
                throw 'Unsupported transform type: ' + transform.type;
            }
        }, this);
      
        // Translate all geometry to their world position
        triangles.forEach(function(triangle, idx) {
            triangles[idx] = this.translate(obj.position[0], obj.position[1], obj.position[2], triangle);
        }, this);
    
        output = output.concat(triangles);
    }, this);

    return output;
}


// Rotates all of the verticies of the given triangle.
Scene.prototype.rotate = function(dx, dy, dz, triangle)
{
    triangle.forEach(function(vertex, index) {
        triangle[index].pos = new Vector(vertex.pos).rotate(dx, dy, dz);
    }, this);    
    return triangle;
}
      
// Scales all of the verticies of the given triangle.
Scene.prototype.scale = function(dx, dy, dz, triangle)
{
    triangle.forEach(function(vertex, index) {
        triangle[index].pos = new Vector(vertex.pos).scale(dx, dy, dz);
    }, this);
    return triangle;
}

// Translates all of the verticies of the given triangle.
Scene.prototype.translate = function(dx, dy, dz, triangle)
{
    triangle.forEach(function(vertex, index) {
        triangle[index].pos = new Vector(vertex.pos).translate(dx, dy, dz);
    }, this);
    return triangle;
}


// Renders the scene as a wireframe
Scene.prototype._renderWireframe = function(vertices, viewport)
{
    viewport.clear('#000');
    
    // Draw each triangle
    for (ii=0; ii<vertices.length; ++ii) {
        var triangle = vertices[ii];
        var a = viewport.mapToViewport(new Vector(triangle[0].pos), 25);
        var b = viewport.mapToViewport(new Vector(triangle[1].pos), 25);
        var c = viewport.mapToViewport(new Vector(triangle[2].pos), 25);

        viewport.drawLine(a.x, a.y, b.x, b.y, '#fff', 1); 
        viewport.drawLine(b.x, b.y, c.x, c.y, '#fff', 1);
        viewport.drawLine(c.x, c.y, a.x, a.y, '#fff', 1);      
    };
}