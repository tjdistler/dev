// Implements the render API


//--------------------------------
// RenderConstants
//
// This data is passed to the shaders as the 'constants' argument.
//--------------------------------
function RenderConstants(viewMatrix)
{
    this.viewMatrix = viewMatrix;
}

RenderConstants.prototype.getViewMatrix = function() {
    return this.viewMatrix;
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
    this.vertShaderCb = null;
    this.vertexShaderCbArgs = null;
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
 * vertShader - TODO
 * vertArgs - A user-defined object to pass to the vertex shader on
 *      each invocation.
 * fragShader - TODO
 * fragArgs - A user-defined object to pass to the fragment shader on
 *      each invocation.
 */
Renderer.prototype.setShaders = function(vertShader, vertArgs, fragShader, fragArgs)
{
    this.vertShader = vertShader;
    this.vertArgs = vertArgs;
    
    //TODO
}

/*
 * Shader callbacks provide a way for the application to get access to data at 
 * intermediate stages in the pipeline.
 *
 * vertexShaderCb - function(triangles) - A function that will be called with the
 *          resulting triangles output from the vertex shader.
 * vertexShaderCbArgs - User defined object to pass to the callback.
 */
Renderer.prototype.setShaderCallbacks = function(vertexShaderCb, vertexShaderCbArgs)
{
    this.vertexShaderCb = vertexShaderCb;
    this.vertexShaderCbArgs = vertexShaderCbArgs;
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
    // Setup render constants for the shaders
    var constants = new RenderConstants(this.camera.viewMatrix);
    
    // Run vertex shader
    var vertexShaderOutput = this.triangles;
    if (this.vertShader) {
        vertexShaderOutput = this._executeVertexShader(this.triangles, this.vertShader, constants, this.vertArgs);
        if (this.vertexShaderCb)
            this.vertexShaderCb(vertexShaderOutput, this.vertexShaderCbArgs);
    }
    
    //TODO - Rasterize
    
    //TODO - Run fragment shader
}


Renderer.prototype._executeVertexShader = function(triangles, shader, constants, args)
{
    var shaderOutput = [];
    
    // Loop over each triangle...
    triangles.forEach(function(triangle, triIndex) {

        logTriangleVertices('Before Vertex Shader[' + triIndex + ']: ', triangle);

        // Each triangle is defined as [Vertex,Vertex,Vertex]. Loop over each vertex...
        var outputTriangle = [];
        triangle.forEach(function(vertex, vertIndex) {
            outputTriangle[vertIndex] = shader(vertex, constants, args);
        });
        
        shaderOutput[triIndex] = outputTriangle;

        logTriangleVertices('After Vertex Shader[' + triIndex + ']: ', outputTriangle);
    });
    
    return shaderOutput
}