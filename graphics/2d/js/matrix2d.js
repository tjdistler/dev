
// Represents a 2D matrix.
// data - A Matrix2d object or an array of the form [[a,b],[c,d]]
function Matrix2d(data)
{
  if (Array.isArray(data))
  {
    assert(data.length == 2, 'Matrix2d data must be of the form [[a,b], [c,d]]');
    this.a = data[0][0];
    this.b = data[0][1];
    this.c = data[1][0];
    this.d = data[1][1];
  }
  else
  {
    this.a = data.a;
    this.b = data.b;
    this.c = data.c;
    this.d = data.d;
  }
}


// Converts the matrix object to an array of the for [[a,b], [c,d]]
Matrix2d.prototype.toArray = function()
{
  return [[this.a, this.b], [this.c, this.d]];
}


// Transforms this matrix by another matrix, in the form "(other)(this)"
// Returns a new Matrix2d.
Matrix2d.prototype.multiply = function(matrix2d)
{
  var result = [[],[]];
  result[0][0] = matrix2d.a * this.a + matrix2d.b * this.c;
  result[0][1] = matrix2d.a * this.b + matrix2d.b * this.d;
  result[1][0] = matrix2d.c * this.a + matrix2d.d * this.c;
  result[1][1] = matrix2d.c * this.b + matrix2d.d * this.d;
  return new Matrix2d(result);
}
