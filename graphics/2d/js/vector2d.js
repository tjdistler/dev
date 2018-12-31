
// Represents a 2D vector.
// data - A Vector2d object or an array of the form [x,y]
function Vector2d(data)
{
  if (Array.isArray(data))
  {
    assert(data.length == 2, 'Vector2d data must be of the form [x,y]');
    this.x = data[0];
    this.y = data[1];
  }
  else
  {
    this.x = data.x;
    this.y = data.y;
  }
}


// Returns the length (magnitude) of the vector
Vector2d.prototype.length = function()
{
  return Math.sqrt(this.x * this.x + this.y * this.y);
}


// Converts the vector to an array of the form [x,y]
Vector2d.prototype.toArray = function()
{
  return [this.x, this.y];
}


// Normalizes (i.e. make it length 1) the vector and returns a new Vector2d.
Vector2d.prototype.normalize = function()
{
  var len = this.length();
  if (len > 0)
  {
    // Calculate the inverse length because multiplication is faster than division
    // in the later step.
    var ilen = 1.0 / len;
    return new Vector2d( [this.x * ilen, this.y * ilen] );
  }
  
  return this;
}


// Calculate the dot product (i.e. project this vector onto the other).
// Note: 
//   1. If both vectors A and B are normalized, then the dot product is the cosine of
//      the angle between them. As such, a product of 0 means A and B are perpendicular.
//      1 means they are facing the same direction and -1 means they are facing the
//      opposite direction.
//   2. If only B is normalized, then A.B = len(A)*cos(angle)
//   3. If neither A nor B is normalized, then A.B = len(A)*len(B)*cos(angle)... or
//      angle = acos( A.B / (len(A)*len(B)) )
Vector2d.prototype.dotProduct = function(vector2d)
{
  if (Array.isArray(vector2d))
  {
    assert(vector2d.length == 2, 'Vector must have 2 elements.');
    return this.x * vector2d[0] + this.y * vector2d[1];
  }
  else
  {
    return this.x * vector2d.x + this.y * vector2d.y;
  }
}


// Transforms this vector by matrix and returns a new Vector2d.
// matrix [[a,b],[c,d]] or [[a,b,c],[d,e,f],[g,h,i]]
Vector2d.prototype.transform = function(matrix)
{
  var m = matrix;
  if (!Array.isArray(matrix))
    m = matrix.toArray();

  var result = [];
  if (m.length == 2)
  {
    result[0] = m[0][0] * this.x + m[0][1] * this.y;
    result[1] = m[1][0] * this.x + m[1][1] * this.y;
    return new Vector2d(result);
  }
  else if (m.length == 3)
  {
    // Assume z of 1
    result[0] = m[0][0] * this.x + m[0][1] * this.y + m[0][2];
    result[1] = m[1][0] * this.x + m[1][1] * this.y + m[1][2];
    return new Vector2d(result);
  }
  
  assert(false, 'Unsupported matrix');
}


// Scales the vector by the specified amount and returns a new Vector2d.
Vector2d.prototype.scale = function(sx, sy)
{
  return this.transform([
    [sx, 0 ],
    [0,  sy]
  ]);
}


// Rotates the vector by the specified angle and returns a new Vector2d.
// angle - degrees.
Vector2d.prototype.rotate = function(angle)
{
  var radians = angle * (Math.PI / 180.0);
  return this.transform([
    [Math.cos(radians), -Math.sin(radians)],
    [Math.sin(radians),  Math.cos(radians)]
  ]);
}


// Shears the vector by the specified amount and returns a new Vector2d.
Vector2d.prototype.shear = function(sx, sy)
{
  return this.transform([
    [1,  sy],
    [sx, 1 ]
  ]);
}


// Translates the vector by the specified amount and returns a new Vector2d.
Vector2d.prototype.translate = function(tx, ty)
{
  return this.transform([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1 ]
  ]);
}


