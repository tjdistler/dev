/* Defines a Camera object
 *
 * position - The Vector position of the camera in world space.
 * lookAt - The Vector point the camera is looking at.
 * fov - The horizontal field of view, in degrees
 * aspectRatio - The viewport aspect ratio (w/h)
 */
function Camera(position, lookAt, fov, aspectRatio)
{
    this.position = position;
    this.lookAt = lookAt;
    this.fov = fov;

    // zoomx   winx
    // ----- = ----
    // zoomy = winy
    
    this.zoomx = 1 / Math.tan( (fov*Math.PI/180) /2);
    this.zoomy = 1/aspectRatio * this.zoomx;
    
    // Calculate view matrix
    var zAxis = position.subtract(lookAt).normalize();
    var xAxis = new Vector([0,1,0]).crossProduct(zAxis).normalize();
    var yAxis = zAxis.crossProduct(xAxis);
    
    var O = new Matrix([    // Orientation matrix
        [xAxis.x, yAxis.x, zAxis.x, 0],
        [xAxis.y, yAxis.y, zAxis.y, 0],
        [xAxis.z, yAxis.z, zAxis.z, 0],
        [0,       0,       0,       1]
    ]);
    
    var T = new Matrix([    // Translation matrix
        [1, 0, 0, -position.x],
        [0, 1, 0, -position.y],
        [0, 0, 1, -position.z],
        [0, 0, 0, 1]
    ]);
    
    this.viewMatrix = O.multiply(T);
}