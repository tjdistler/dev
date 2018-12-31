// Represents a scene to render. A scene contains all the information needed
// to render and animate the ojects in the scene.
//
// data - The scene object to render
function Scene(scene)
{
  this.scene = scene;
  this.fps = this.scene.fps;
  this.duration = this.scene.duration;
}


// Returns an array of triangles to be rendered. This function performs all
// vertex transformations into world space.
// sceneTime - ms since the beginning of the scene.
// Return value in the form [ [[x,y],[x,y],[x,y]], [...], ... ]
Scene.prototype.generateFrameGeometry = function(sceneTime)
{
  var output = [];

  this.scene.objects.forEach(function(obj, objIdx) {

    // Check to see if the object exists during this sceneTime.
    // Undefined begin/end times mean 'always exists'.
    if (obj.begin_ts !== undefined && sceneTime < obj.begin_ts)
      return;
    if (obj.end_ts !== undefined && sceneTime > obj.end_ts)
      return;

    // All transformations happen on a copy of the object geometry.
    var triangles = deepCopy(obj.triangles);

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
        var angle = transform.angle * transformPercentage;
        triangles.forEach(function(triangle, idx) {
          triangles[idx] = this.rotate(angle, triangle);
        }, this);
      }
      else if (transform.type == 'scale')
      {
        // Note: scaling starts at 1, not 0
        var magnitudeX = transform.dx - 1;
        var magnitudeY = transform.dy - 1;
        var dx = 1 + (magnitudeX * transformPercentage);
        var dy = 1 + (magnitudeY * transformPercentage);

        triangles.forEach(function(triangle, idx) {
          triangles[idx] = this.scale(dx, dy, triangle);
        }, this);
      }
      else if (transform.type == 'translate')
      {
        var dx = transform.dx * transformPercentage;
        var dy = transform.dy * transformPercentage;

        triangles.forEach(function(triangle, idx) {
          triangles[idx] = this.translate(dx, dy, triangle);
        }, this);
      }
      else
      {
          throw 'Unsupported transform type: ' + transform.type;
      }
    }, this);
      
    // Translate all geometry to their world position
    triangles.forEach(function(triangle, idx) {
      triangles[idx] = this.translate(obj.position[0], obj.position[1], triangle);
    }, this);
    
    output = output.concat(triangles);
  }, this);

  return output;
}


// Rotates all of the verticies of the given triangle.
Scene.prototype.rotate = function(angle, triangle)
{
  var result = [];
  triangle.forEach(function(vertex) {
    result.push(new Vector2d(vertex).rotate(angle).toArray());
  }, this);
  return result;
}
      
// Scales all of the verticies of the given triangle.
Scene.prototype.scale = function(dx, dy, triangle)
{
  var result = [];
  triangle.forEach(function(vertex) {
    result.push(new Vector2d(vertex).scale(dx, dy).toArray());
  }, this);
  return result;
}

// Translates all of the verticies of the given triangle.
Scene.prototype.translate = function(dx, dy, triangle)
{
  var result = [];
  triangle.forEach(function(vertex) {
    result.push(new Vector2d(vertex).translate(dx, dy).toArray());
  }, this);
  return result;
}