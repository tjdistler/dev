function VectorTests()
{
    this.a1 = [1, 2, 3];
    this.a2 = [1, 2, 3, 1];
    this.a3 = [9.1, 8.2, 7.3, 6.4];
    
    this.a1Len = 3.7417;
    this.a2Len = 3.8730;
    this.a3Len = 15.6301;
    
    this.v1dotv2 = 14;
    this.v2dotv3 = 53.8;
    
    this.PRECISION = 4; //decimal places
}


VectorTests.prototype.run = function()
{
    console.log('Executing VectorTests...');
 
    this.TestConstructor();
    this.TestLength();
    this.TestToArray();
    this.TestNormalize();
    this.TestTransform();
    this.TestScale();
    this.TestRotate();
    this.TestTranslate();
    this.TestDotProduct();
    this.TestCrossProduct();
}


VectorTests.prototype.TestConstructor = function()
{
    console.log('\tTestConstructor');
    
    var v1 = new Vector(this.a1);
    assert(this.equals(v1, this.a1), 'Vector doesn\'t match original array');

    var v2 = new Vector(this.a2);
    assert(this.equals(v2, this.a2), 'Vector doesn\'t match original array');
    
    var v1a = new Vector(v1);
    var v2a = new Vector(v2);
    
    assert(this.equals(v1, v1a), 'Copied vector doesn\'t match original');
    assert(this.equals(v2, v2a), 'Copied vector doesn\'t match original');
    assert(!this.equals(v1, v2), 'v1 and v2 should not be equal');
}


VectorTests.prototype.TestLength = function()
{
    console.log('\tTestLength');
    
    var v1 = new Vector(this.a1);
    var v2 = new Vector(this.a2);
    var v3 = new Vector(this.a3);
    
    assert(round(v1.length(), this.PRECISION) == this.a1Len, 'Vector length incorrect');
    assert(round(v2.length(), this.PRECISION) == this.a2Len, 'Vector length incorrect');
    assert(round(v3.length(), this.PRECISION) == this.a3Len, 'Vector length incorrect');
    
    var v1a = new Vector(v1);
    assert(round(v1a.length(), this.PRECISION) == this.a1Len, 'Vector length incorrect');
    assert(round(v1a.length(), this.PRECISION) == round(v1.length(), this.PRECISION), 'Vector length incorrect');
}


VectorTests.prototype.TestToArray = function()
{
    console.log('\tTestToArray');
    
    var v1 = new Vector(this.a1);
    var v2 = new Vector(this.a2);
    var v3 = new Vector(this.a3);
    
    // Define array comparison function
    var arrayEqual = function(a,b) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length != b.length)
            return false;
        for (i=0; i<a.length; ++i)
        {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }
    
    assert(arrayEqual(v1.toArray(), [this.a1[0], this.a1[1], this.a1[2], 0]), 'The returned array is incorrect');
    assert(arrayEqual(v2.toArray(), this.a2), 'The returned array is incorrect');
    assert(arrayEqual(v3.toArray(), this.a3), 'The returned array is incorrect');
}


VectorTests.prototype.TestNormalize = function()
{
    console.log('\tTestNormalize');
    
    var v1 = new Vector(this.a1);
    var v2 = new Vector(this.a2);
    var v3 = new Vector(this.a3);
    
    var v1Norm = [
        v1.x / v1.length(),
        v1.y / v1.length(),
        v1.z / v1.length(),
        v1.w / v1.length()
    ];
    var v2Norm = [
        v2.x / v2.length(),
        v2.y / v2.length(),
        v2.z / v2.length(),
        v2.w / v2.length()
    ];
    var v3Norm = [
        v3.x / v3.length(),
        v3.y / v3.length(),
        v3.z / v3.length(),
        v3.w / v3.length()
    ];
        
    assert(this.equals(v1.normalize(), v1Norm), 'The normalized vector is incorrect');
    assert(this.equals(v2.normalize(), v2Norm), 'The normalized vector is incorrect');
    assert(this.equals(v3.normalize(), v3Norm), 'The normalized vector is incorrect');
}


VectorTests.prototype.TestTransform = function()
{
    console.log('\tTestTransform');
    
    var v1 = new Vector(this.a1);    
    var M1 = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ];
    var M1Result = [46, 28, 10, 0];
    
    assert(this.equals(v1.transform(M1), M1Result), 'Transform result incorrect');
    assert(this.equals(v1.transform(new Matrix(M1)), M1Result), 'Transform result incorrect');
    
    var v3 = new Vector(this.a3);
    var M2 = [
        [9, 8, 7, 6],
        [5, 4, 3, 2],
        [1, 0, 9, 8],
        [7, 6, 5, 4]
    ];
    var M2Result = [237, 112.999999, 126, 174.999999];
    assert(this.equals(v3.transform(M2), M2Result), 'Transform result incorrect');
    assert(this.equals(v3.transform(new Matrix(M2)), M2Result), 'Transform result incorrect');
}


VectorTests.prototype.TestScale = function()
{
    console.log('\tTestScale');

    var v2 = new Vector(this.a2);
    var v3 = new Vector(this.a3);
    
    var v2Scaled = [
        v2.x * 9,
        v2.y * 8,
        v2.z * 7,
        v2.w
    ];
    var v3Scaled = [
        v3.x * 6,
        v3.y * 5,
        v3.z * 4,
        v3.w
    ];

    assert(this.equals(v2.scale(9,8,7), v2Scaled), 'Scaled result incorrect');
    assert(this.equals(v3.scale(6,5,4), v3Scaled), 'Scaled result incorrect');
}


VectorTests.prototype.TestRotate = function()
{
    console.log('\tTestRotate');
    
    var v1 = new Vector([0, 0, 1]);
    var v2 = new Vector([1, 0, 0]);
    
    var v1Rx90   = [0, -1, 0];            //Rotate around x-axis 90
    var v1Rx45   = [0, -0.7071, 0.7071];  //Rotate around x-axis 45
    var v1Rym90  = [-1, 0, 0];           //Rotate around y-axis -90
    var v1Ry270  = [-1, 0, 0];           //Rotate around y-axis 270
    var v1Rym135 = [-0.7071, 0, -0.7071];//Rotate around y-axis -135
    var v1Rz30   = v1;                   //Rotate around z-axis 30 (no change)
    
    var v2Rz30   = [0.8660, 0.5, 0];     //Rotate around z-axis 30
    var v2Rzm90  = [0, -1, 0];           //Rotate around z-axis -90
    
    assert(this.equals(v1.rotate(90,0,0), v1Rx90), 'Rotation result incorrect');
    assert(this.equals(v1.rotate(45,0,0), v1Rx45), 'Rotation result incorrect');
    assert(this.equals(v1.rotate(0,-90,0), v1Rym90), 'Rotation result incorrect');
    assert(this.equals(v1.rotate(0,270,0), v1Ry270), 'Rotation result incorrect');
    assert(this.equals(v1.rotate(0,-135,0), v1Rym135), 'Rotation result incorrect');
    assert(this.equals(v1.rotate(0,0,30), v1Rz30), 'Rotation result incorrect');
    
    assert(this.equals(v2.rotate(0,0,30), v2Rz30), 'Rotation result incorrect');
    assert(this.equals(v2.rotate(0,0,-90), v2Rzm90), 'Rotation result incorrect');
    
    // Mulit-axis rotation
    var v1Multi = [0.7071, 0, -0.7071];
    assert(this.equals(v1.rotate(45, -180, 90), v1Multi), 'Rotation result incorrect');
}


VectorTests.prototype.TestTranslate = function()
{
    console.log('\tTestTranslate');
    
    var v2 = new Vector(this.a2);
    var v3 = new Vector([9.1, 8.2, 7.3, 1]);
    
    var v2Trans = [
        v2.x + 9,
        v2.y - 8,
        v2.z + 7,
        v2.w
    ];
    var v3Trans = [
        v3.x + 1.1,
        v3.y + 0,
        v3.z + -2.2,
        v3.w
    ];

    assert(this.equals(v2.translate(9, -8, 7), v2Trans), 'Translation result incorrect');
    assert(this.equals(v3.translate(1.1, 0, -2.2), v3Trans), 'Translation result incorrect');
}


VectorTests.prototype.TestDotProduct = function()
{
    console.log('\tTestDotProduct');
    
    var v1 = new Vector(this.a1);
    var v2 = new Vector(this.a2);
    var v3 = new Vector(this.a3);
    
    assert(v1.dotProduct(v2) == this.v1dotv2, 'DotProduct result incorrect');
    assert(v2.dotProduct(v3) == this.v2dotv3, 'DotProduct result incorrect');

    var v4 = new Vector([1, 0, 0]);
    var v5 = new Vector([0, 1, 0]);
    var v6 = new Vector([0.7071, 0.7071, 0]);
    var d45 = 45 * Math.PI/180;

    // Verify the dot product of 2 normalized vectors equals the cosine of the angle
    // between them.
    assert(round(v4.dotProduct(v6), this.PRECISION) == round(Math.cos(d45), this.PRECISION), 'DotProduct not equal to cosine of angle');
    
    // Verify other properties
    assert(v4.dotProduct(v4) == 1, 'DotProduct parallel test failed');
    assert(v4.dotProduct(v4.scale(-1,1,1)) == -1, 'DotProduct inverse direction test failed');
    assert(v4.dotProduct(v5) == 0, 'DotProduct perpendicular test failed');
}


VectorTests.prototype.TestCrossProduct = function()
{
    console.log('\tTestCrossProduct');
    
    var v1 = new Vector([1, 0, 0]);
    var v2 = new Vector([0, 1, 0]);
    var v3 = new Vector([0.7071, 0, 0.7071]);
    var v4 = new Vector([2, 0, -4]);
    
    var v1v2 = new Vector([0, 0, 1]);
    var v3v4 = new Vector([0, 1, 0]);
    
    assert(this.equals(v1.crossProduct(v2), v1v2), 'CrossProduct result incorrect');
    assert(this.equals(v3.crossProduct(v4).normalize(), v3v4), 'CrossProduct result incorrect');
    
    // Verify the equality: ||axb|| = ||a||*||b||*sin(t)
    var a = new Vector([0, 1, 0]);
    var b = new Vector([0.7071, 0.7071, 0]); // 45 degree angle from v5
    
    var axbLen = round( a.crossProduct(b).length(), this.PRECISION );
    var abLenSin = round( a.length() * b.length() * Math.sin(45*Math.PI/180), this.PRECISION );
    
    assert( axbLen == abLenSin, 'CrossProduct result incorrect');
}



// HELPER FUNCTIONS

VectorTests.prototype.equals = function(v1, v2)
{
    var a = v1;
    var b = v2;
    if (!Array.isArray(v1))
        a = v1.toArray();
    if (!Array.isArray(v2))
        b = v2.toArray();
    
    assert(a.length >= 3, 'v1 does not represent a Vector');
    assert(b.length >= 3, 'v2 does not represent a Vector');
    
    // Limit precision first
    a.forEach(function(v,idx) {
        a[idx] = round(v,this.PRECISION);
    }.bind(this));
    
    b.forEach(function(v,idx) {
        b[idx] = round(v,this.PRECISION);
    }.bind(this));
    
    // Compare x,y,z
    if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2])
        return false;
    
    // Compare optional 'w' value
    if (a.length == 4 && b.length == 4)
        return a[3] == b[3];
    if (a.length == 4 && b.length == 3)
        return a[3] == 0;
    
    return b[3] == 0;
}