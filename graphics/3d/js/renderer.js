// Implements the render API


//--------------------------------
// Render State
//
// This data is passed to the shaders as the 'renderer' argument.
//--------------------------------
function RenderState()
{
    //TODO
}


//--------------------------------
// Renderer
//--------------------------------
function Renderer()
{
    this.reset();
}


Renderer.prototype.reset = function()
{
    // Options
    this.rasterize = true;
    
    this.triangles = null;
    this.lights = null;
    this.camera = null;
    this.vertShader = null;
    this.vertArgs = null;
    this.fragShader = null;
    this.fragArgs = null;
    this.textures = null;
}


/* Sets the vertices for all the triangles in the scene.
 * All vertices are assumed to be in world-space.
 * vertices - Array of triangle Vertex objects of the form:
 *   [[Vertex,Vertex,Vertex], ...]
 */
Renderer.prototype.setVertices = function(triangles)
{
    this.triangles = triangles;
}


Renderer.prototype.setLights = function(lights)
{
    this.lights = lights;
}


Renderer.prototype.setCamera = function(camera)
{
    this.camera = camera;
}


/* Set the vertex and fragment shaders for rendering.
 *
 * vertShader - A function that will be called for every vertex.
 *      The function must have the form function(vertex[3], args, state),
 *      where 'vertex' is the array of Vertex of a triangle to operate on,
 *      'args' is the 'vertArgs' passed into the setShaders call, and
 *      'state' is a RenderState object for interacting with the render
 *      engine. The shader function must a valid Vertex array.
 * vertArgs - A user-defined object to pass to the vertex shader on
 *      each invocation.
 * fragShader - A function that will be called for every fragment.
 *      The function must have the form function(fragment, args, state),
 *      where 'fragment' is the Fragment to operate on, 'args' is the
 *      'fragArgs' passed into the setShaders call, and 'state' is a
 *      RenderState object for interacting with the render engine. The
 *      shader function must return an array of Color objects, where the
 *      first Color object will be written to buffer[0] of the execute()
 *      call, etc. If the returned Color array is shorter than the buffer
 *      list, then only the first n entries will be written. If an entry in
 *      the Color array is null, then that buffer will be skipped.
 * fragArgs - A user-defined object to pass to the fragment shader on
 *      each invocation.
 */
Renderer.prototype.setShaders = function(vertShader, vertArgs, fragShader, fragArgs)
{
    //TODO
}


Renderer.prototype.setTextures = function(textures)
{
    //TODO
}


/* Excutes the render pipeline, using the specified render options, to
 * render the scene into the provided buffers.
 *
 * options - The render options to use. Supported options are:
 *      {
 *          rasterize: true // if false, processing will stop after clipping/culling
 *      }
 * buffers - The array of buffers to use during rendering. The size,
 *      content, and use of the buffers is user-defined. The values written
 *      to the buffers are defined by the shaders.
 */
Renderer.prototype.execute = function(options, buffers)
{
    this._transformToViewSpace();
    
    //TODO
}


/*
 * Transforms all the vertices to view space.
 */
Renderer.prototype._transformToViewSpace = function()
{
    this.triangles.forEach(function(triangle, index) {
        this.triangles[index][0].pos = triangle[0].pos.transform(this.camera.viewMatrix);
        this.triangles[index][1].pos = triangle[1].pos.transform(this.camera.viewMatrix);
        this.triangles[index][2].pos = triangle[2].pos.transform(this.camera.viewMatrix);
    }, this);
}