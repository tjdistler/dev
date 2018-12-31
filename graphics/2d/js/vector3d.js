// Represents a #D vector.
// data - A Vector3d object or an array of the form [x,y,3]
function Vector3d(data)
{
  if (Array.isArray(data))
  {
    assert(data.length == 3, 'Vector3d data must be of the form [x,y,z]');
    this.x = data[0];
    this.y = data[1];
    this.z = data[2];
  }
  else
  {
    this.x = data.x;
    this.y = data.y;
    this.z = data.z;
  }
}



// Calculate the cross product (i.e. finds the vector perpendicular to the plane
// defined by A and B). Returns a new Vector3d.
Vector3d.prototype.crossProduct = function(vector3d)
{
  if (Array.isArray(vector3d))
  {
    assert(vector3d.length == 3, 'Vector must have 3 elements.');
    return new Vector3d([
      this.y*vector3d[2] - this.z*vector3d[1],
      this.z*vector3d[0] - this.x*vector3d[2],
      this.x*vector3d[1] - this.y*vector3d[0]
    ]);
  }
  else
  {
    return new Vector3d([
      this.y*vector3d.z - this.z*vector3d.y,
      this.z*vector3d.x - this.x*vector3d.z,
      this.x*vector3d.y - this.y*vector3d.x
    ]);
  }
}
