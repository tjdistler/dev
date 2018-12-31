function MatrixTests()
{
    this.A1 = [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ];
    
    this.A1a = [ // 4x4 version of A1
        [1,2,3,0],
        [4,5,6,0],
        [7,8,9,0],
        [0,0,0,1]
    ];

    this.A2 = [
        [1,2,3,4],
        [5,6,7,8],
        [9,0,1,2],
        [3,4,5,6]
    ];
    
    this.A1A2 = [
        [38, 14, 20, 26],
        [83, 38, 53, 68],
        [128, 62, 86, 110],
        [3, 4, 5, 6]
    ];
    
    this.A2A1 = [
        [30, 36, 42, 4],
        [78, 96, 114, 8],
        [16, 26, 36, 2],
        [54, 66, 78,6]
    ];
}


MatrixTests.prototype.run = function()
{
    console.log('Executing MatrixTests...');
 
    this.TestConstructor();
    this.TestMultiply();
    this.TestToArray();
}


MatrixTests.prototype.TestConstructor = function()
{
    console.log('\tTestConstructor');
    
    var M1 = new Matrix(this.A1);
    assert(this.equals(M1, this.A1), 'Matrix doesn\'t match original array');

    var M2 = new Matrix(this.A2);
    assert(this.equals(M2, this.A2), 'Matrix doesn\'t match original array');
    
    var M1a = new Matrix(M1);
    var M2a = new Matrix(M2);
    
    assert(this.equals(M1, M1a), 'Copied matrix doesn\'t match original');
    assert(this.equals(M2, M2a), 'Copied matrix doesn\'t match original');
    assert(!this.equals(M1, M2), 'M1 and M2 should not be equal');
}


MatrixTests.prototype.TestMultiply = function()
{
    console.log('\tTestMultiply');
    
    var M1 = new Matrix(this.A1);
    var M2 = new Matrix(this.A2);
    
    assert(this.equals(M1.multiply(M2), this.A1A2), 'M1*M2 result doesn\'t match expected value');
    assert(this.equals(M1.multiply(this.A2), this.A1A2), 'M1*A2 result doesn\'t match expected value');
    
    assert(this.equals(M2.multiply(M1), this.A2A1), 'M2*M1 result doesn\'t match expected value');
    assert(this.equals(M2.multiply(this.A1a), this.A2A1), 'M2*A1 result doesn\'t match expected value');
    
    // Order matters
    assert(!this.equals(M1.multiply(M2), this.A2A1), 'M1*M2 result doesn\'t match expected value');
    assert(!this.equals(M2.multiply(M1), this.A1A2), 'M2*M1 result doesn\'t match expected value');
}


MatrixTests.prototype.TestToArray = function()
{
    console.log('\tTestToArray');

    var M1 = new Matrix(this.A1);
    var M2 = new Matrix(this.A2);
    
    // toArray should always return a 4x4 matrix array
    assert(M1.toArray().length == 4, 'M1 length is incorrect');
    assert(M2.toArray().length == 4, 'M2 length is incorrect');
    
    assert(M1.toArray()[0].length == 4, 'M1[0] length is incorrect');
    assert(M1.toArray()[1].length == 4, 'M1[1] length is incorrect');
    assert(M1.toArray()[2].length == 4, 'M1[2] length is incorrect');
    assert(M1.toArray()[3].length == 4, 'M1[3] length is incorrect');
}



// HELPER FUNCTIONS

// Returns true if the matrices are equal. M2 can be an array or Matrix.
MatrixTests.prototype.equals = function(M1, M2)
{
    if (Array.isArray(M2))
    {
        if (M1.rowA[0] != M2[0][0]) return false;
        if (M1.rowA[1] != M2[0][1]) return false;
        if (M1.rowA[2] != M2[0][2]) return false;
        if (M2[0].length > 3)
            if (M1.rowA[3] != M2[0][3]) return false;
        
        if (M1.rowB[0] != M2[1][0]) return false;
        if (M1.rowB[1] != M2[1][1]) return false;
        if (M1.rowB[2] != M2[1][2]) return false;
        if (M2[1].length > 3)
            if (M1.rowB[3] != M2[1][3]) return false;
        
        if (M1.rowC[0] != M2[2][0]) return false;
        if (M1.rowC[1] != M2[2][1]) return false;
        if (M1.rowC[2] != M2[2][2]) return false;
        if (M2[2].length > 3)
            if (M1.rowC[3] != M2[2][3]) return false;
        
        if (M2.length > 3)
        {
            if (M1.rowD[0] != M2[3][0]) return false;
            if (M1.rowD[1] != M2[3][1]) return false;
            if (M1.rowD[2] != M2[3][2]) return false;
            if (M1.rowD[3] != M2[3][3]) return false;
        }
        
        return true;
    }
    else
    {
        if (M1.rowA.length != M2.rowA.length ||
            M1.rowB.length != M2.rowB.length ||
            M1.rowC.length != M2.rowC.length ||
            M1.rowD.length != M2.rowD.length)
        {
            return false;
        }

        var equal = true;
        M1.rowA.forEach(function(v1, index) {
            if (v1 != M2.rowA[index])
                equal = false;
        });
        M1.rowB.forEach(function(v1, index) {
            if (v1 != M2.rowB[index])
                equal = false;
        });
        M1.rowC.forEach(function(v1, index) {
            if (v1 != M2.rowC[index])
                equal = false;
        });
        M1.rowD.forEach(function(v1, index) {
            if (v1 != M2.rowD[index])
                equal = false;
        });

        return equal;
    }
}