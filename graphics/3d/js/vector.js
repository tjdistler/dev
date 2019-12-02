// Represents a 3D vector.
// data - A Vector object or an array of the form [x,y,z] or [x,y,z,w]
//
// NOTE: Assumes left-hand coordinate system
function Vector(data)
{
    if (Array.isArray(data))
    {
        if (data.length == 3)
        {
            this.x = data[0];
            this.y = data[1];
            this.z = data[2];
            this.w = 0;
        }
        else if (data.length == 4)
        {
            this.x = data[0];
            this.y = data[1];
            this.z = data[2];
            this.w = data[3];
        }
        else
            assert(false, 'Unsupported vector array size');
    }
    else
    {
        this.x = data.x;
        this.y = data.y;
        this.z = data.z;
        this.w = data.w;
    }
}


// Returns the length (magnitude) of the vector
Vector.prototype.length = function()
{
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
}


// Converts the vector to an array of the form [x,y,z,w]
Vector.prototype.toArray = function()
{
    return [this.x, this.y, this.z, this.w];
}


// Subtracts a vector from this vector. Returns a new Vector.
Vector.prototype.subtract = function(other)
{
    if (Array.isArray(other))
    {
        if (other.length == 3)
        {
            return new Vector([this.x-other[0], this.y-other[1], this.z-other[2], this.w]);
        }
        else if (other.length == 4)
        {
            return new Vector([this.x-other[0], this.y-other[1], this.z-other[2], this.w-other[3]]);
        }
        else
            assert(false, 'Unsupported vector array size');
    }
    else
        return new Vector([this.x-other.x, this.y-other.y, this.z-other.z, this.w-other.w]);
}


// Normalizes (i.e. make it length 1) the vector and returns a new Vector.
Vector.prototype.normalize = function()
{
    var len = this.length();
    if (len > 0)
    {
        // Calculate the inverse length because multiplication is faster than division
        // in the later step.
        var ilen = 1.0 / len;
        return new Vector( [this.x * ilen, this.y * ilen, this.z * ilen, this.w * ilen] );
    }
  
    return this;
}


// Transforms this vector by matrix and returns a new Vector.
// matrix [[a,b,c],[d,e,f],[g,h,i]] or [[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]]
Vector.prototype.transform = function(matrix)
{
    var m = matrix;
    if (!Array.isArray(matrix))
        m = matrix.toArray();

    var result = [];
    if (m.length == 3)
    {
        result[0] = m[0][0] * this.x + m[0][1] * this.y + m[0][2] * this.z;
        result[1] = m[1][0] * this.x + m[1][1] * this.y + m[1][2] * this.z;
        result[2] = m[2][0] * this.x + m[2][1] * this.y + m[2][2] * this.z;
        return new Vector(result);
    }
    else if (m.length == 4)
    {
        result[0] = m[0][0] * this.x + m[0][1] * this.y + m[0][2] * this.z + m[0][3] * this.w;
        result[1] = m[1][0] * this.x + m[1][1] * this.y + m[1][2] * this.z + m[1][3] * this.w;
        result[2] = m[2][0] * this.x + m[2][1] * this.y + m[2][2] * this.z + m[2][3] * this.w;
        result[3] = m[3][0] * this.x + m[3][1] * this.y + m[3][2] * this.z + m[3][3] * this.w;
        
        return new Vector(result);
    }
  
    assert(false, 'Unsupported matrix');
}


// Scales the vector by the specified amount and returns a new Vector.
Vector.prototype.scale = function(sx, sy, sz)
{
    return this.transform([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1]
    ]);
}


// Rotates the vector by the specified angles and returns a new Vector.
// ax - The angle of rotation about the x axis (in degrees)
// ay - The angle of rotation about the y axis
// az - The angle of rotation about the z axis
Vector.prototype.rotate = function(ax, ay, az)
{
    // Convert to radians
    iPI = Math.PI / 180.0;
    var rx = ax * iPI;
    var ry = ay * iPI;
    var rz = az * iPI;
    
    var sinrx = Math.sin(rx);
    var sinry = Math.sin(ry);
    var sinrz = Math.sin(rz);
    var cosrx = Math.cos(rx);
    var cosry = Math.cos(ry);
    var cosrz = Math.cos(rz);
    
    var dx = this.transform([
        [1, 0,      0,     0],
        [0, cosrx, -sinrx, 0],
        [0, sinrx,  cosrx, 0],
        [0, 0,      0,     1]
    ]);
    
    var dy = dx.transform([
        [ cosry, 0, sinry, 0],
        [ 0,     1, 0,     0],
        [-sinry, 0, cosry, 0],
        [ 0,     0, 0,     1]
    ]);
    
    return dy.transform([
        [cosrz, -sinrz, 0, 0],
        [sinrz,  cosrz, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ]);
}



// Translates the vector by the specified amount and returns a new Vector.
Vector.prototype.translate = function(tx, ty, tz)
{
    return this.transform([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1 ]
    ]);
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
Vector.prototype.dotProduct = function(vector)
{
    if (Array.isArray(vector))
    {
        if (vector.length == 3)
        {
            return this.x * vector[0] + this.y * vector[1] + this.z * vector[2];
        }
        else if (vector.length == 4)
        {
            return this.x * vector[0] + this.y * vector[1] + this.z * vector[2] + this.w * vector[3];
        }
        else
            assert(false, 'Unsupported vector array size');
    }
    else
    {
        return this.x * vector.x + this.y * vector.y + this.z * vector.z + this.w * vector.w;
    }
}


// Calculate the cross product (i.e. finds the vector perpendicular to the plane
// defined by vectors A and B). Returns a new Vector.
Vector.prototype.crossProduct = function(vector)
{
    if (Array.isArray(vector))
    {
        assert(vector.length == 3, 'Vector must have 3 elements.');
        return new Vector([
            this.y * vector[2] - this.z * vector[1],
            this.z * vector[0] - this.x * vector[2],
            this.x * vector[1] - this.y * vector[0]
        ]);
    }
    else
    {
        return new Vector([
            this.y * vector.z - this.z * vector.y,
            this.z * vector.x - this.x * vector.z,
            this.x * vector.y - this.y * vector.x
        ]);
    }
}
