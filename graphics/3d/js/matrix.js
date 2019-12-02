
// Represents a 4D matrix.
// data - An array of one of the forms:
//   [[a,b,c],[d,e,f],[g,h,i]]
//     or
//   [[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]]
//
function Matrix(data)
{
    if (Array.isArray(data))
    {
        if (data.length == 3)
        {
            this.rowA = [data[0][0], data[0][1], data[0][2], 0];
            this.rowB = [data[1][0], data[1][1], data[1][2], 0];
            this.rowC = [data[2][0], data[2][1], data[2][2], 0];
            this.rowD = [0,          0,          0,          1];
        }
        else if (data.length == 4)
        {
            this.rowA = [data[0][0], data[0][1], data[0][2], data[0][3]];
            this.rowB = [data[1][0], data[1][1], data[1][2], data[1][3]];
            this.rowC = [data[2][0], data[2][1], data[2][2], data[2][3]];
            this.rowD = [data[3][0], data[3][1], data[3][2], data[3][3]];
        }
        else
            assert(false, 'Unsupported matrix dimension');
    }
    else
    {
        this.rowA = data.rowA;
        this.rowB = data.rowB;
        this.rowC = data.rowC;
        this.rowD = data.rowD;
    }
}


// Converts the matrix object to an array of the form:
//  [[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]]
Matrix.prototype.toArray = function()
{
    return [this.rowA, this.rowB, this.rowC, this.rowD];
}


// Transforms this matrix by another matrix, in the form "(this)(other)"
// Returns a new Matrix.
Matrix.prototype.multiply = function(matrix)
{
    if (Array.isArray(matrix))
    {
        assert(matrix.length == 4, 'Matrix array width must be 4');
        var result = [];
        result[0] = [
            this.rowA[0] * matrix[0][0] + this.rowA[1] * matrix[1][0] + this.rowA[2] * matrix[2][0] + this.rowA[3] * matrix[3][0],
            this.rowA[0] * matrix[0][1] + this.rowA[1] * matrix[1][1] + this.rowA[2] * matrix[2][1] + this.rowA[3] * matrix[3][1],
            this.rowA[0] * matrix[0][2] + this.rowA[1] * matrix[1][2] + this.rowA[2] * matrix[2][2] + this.rowA[3] * matrix[3][2],
            this.rowA[0] * matrix[0][3] + this.rowA[1] * matrix[1][3] + this.rowA[2] * matrix[2][3] + this.rowA[3] * matrix[3][3]
        ];
        result[1] = [
            this.rowB[0] * matrix[0][0] + this.rowB[1] * matrix[1][0] + this.rowB[2] * matrix[2][0] + this.rowB[3] * matrix[3][0],
            this.rowB[0] * matrix[0][1] + this.rowB[1] * matrix[1][1] + this.rowB[2] * matrix[2][1] + this.rowB[3] * matrix[3][1],
            this.rowB[0] * matrix[0][2] + this.rowB[1] * matrix[1][2] + this.rowB[2] * matrix[2][2] + this.rowB[3] * matrix[3][2],
            this.rowB[0] * matrix[0][3] + this.rowB[1] * matrix[1][3] + this.rowB[2] * matrix[2][3] + this.rowB[3] * matrix[3][3]
        ];
        result[2] = [
            this.rowC[0] * matrix[0][0] + this.rowC[1] * matrix[1][0] + this.rowC[2] * matrix[2][0] + this.rowC[3] * matrix[3][0],
            this.rowC[0] * matrix[0][1] + this.rowC[1] * matrix[1][1] + this.rowC[2] * matrix[2][1] + this.rowC[3] * matrix[3][1],
            this.rowC[0] * matrix[0][2] + this.rowC[1] * matrix[1][2] + this.rowC[2] * matrix[2][2] + this.rowC[3] * matrix[3][2],
            this.rowC[0] * matrix[0][3] + this.rowC[1] * matrix[1][3] + this.rowC[2] * matrix[2][3] + this.rowC[3] * matrix[3][3]
        ];
        result[3] = [
            this.rowD[0] * matrix[0][0] + this.rowD[1] * matrix[1][0] + this.rowD[2] * matrix[2][0] + this.rowD[3] * matrix[3][0],
            this.rowD[0] * matrix[0][1] + this.rowD[1] * matrix[1][1] + this.rowD[2] * matrix[2][1] + this.rowD[3] * matrix[3][1],
            this.rowD[0] * matrix[0][2] + this.rowD[1] * matrix[1][2] + this.rowD[2] * matrix[2][2] + this.rowD[3] * matrix[3][2],
            this.rowD[0] * matrix[0][3] + this.rowD[1] * matrix[1][3] + this.rowD[2] * matrix[2][3] + this.rowD[3] * matrix[3][3]
        ];
        return new Matrix(result);
    }
    else
    {
        var result = [];
        result[0] = [
            this.rowA[0] * matrix.rowA[0] + this.rowA[1] * matrix.rowB[0] + this.rowA[2] * matrix.rowC[0] + this.rowA[3] * matrix.rowD[0],
            this.rowA[0] * matrix.rowA[1] + this.rowA[1] * matrix.rowB[1] + this.rowA[2] * matrix.rowC[1] + this.rowA[3] * matrix.rowD[1],
            this.rowA[0] * matrix.rowA[2] + this.rowA[1] * matrix.rowB[2] + this.rowA[2] * matrix.rowC[2] + this.rowA[3] * matrix.rowD[2],
            this.rowA[0] * matrix.rowA[3] + this.rowA[1] * matrix.rowB[3] + this.rowA[2] * matrix.rowC[3] + this.rowA[3] * matrix.rowD[3]
        ];
        result[1] = [
            this.rowB[0] * matrix.rowA[0] + this.rowB[1] * matrix.rowB[0] + this.rowB[2] * matrix.rowC[0] + this.rowB[3] * matrix.rowD[0],
            this.rowB[0] * matrix.rowA[1] + this.rowB[1] * matrix.rowB[1] + this.rowB[2] * matrix.rowC[1] + this.rowB[3] * matrix.rowD[1],
            this.rowB[0] * matrix.rowA[2] + this.rowB[1] * matrix.rowB[2] + this.rowB[2] * matrix.rowC[2] + this.rowB[3] * matrix.rowD[2],
            this.rowB[0] * matrix.rowA[3] + this.rowB[1] * matrix.rowB[3] + this.rowB[2] * matrix.rowC[3] + this.rowB[3] * matrix.rowD[3]
        ];
        result[2] = [
            this.rowC[0] * matrix.rowA[0] + this.rowC[1] * matrix.rowB[0] + this.rowC[2] * matrix.rowC[0] + this.rowC[3] * matrix.rowD[0],
            this.rowC[0] * matrix.rowA[1] + this.rowC[1] * matrix.rowB[1] + this.rowC[2] * matrix.rowC[1] + this.rowC[3] * matrix.rowD[1],
            this.rowC[0] * matrix.rowA[2] + this.rowC[1] * matrix.rowB[2] + this.rowC[2] * matrix.rowC[2] + this.rowC[3] * matrix.rowD[2],
            this.rowC[0] * matrix.rowA[3] + this.rowC[1] * matrix.rowB[3] + this.rowC[2] * matrix.rowC[3] + this.rowC[3] * matrix.rowD[3]
        ];
        result[3] = [
            this.rowD[0] * matrix.rowA[0] + this.rowD[1] * matrix.rowB[0] + this.rowD[2] * matrix.rowC[0] + this.rowD[3] * matrix.rowD[0],
            this.rowD[0] * matrix.rowA[1] + this.rowD[1] * matrix.rowB[1] + this.rowD[2] * matrix.rowC[1] + this.rowD[3] * matrix.rowD[1],
            this.rowD[0] * matrix.rowA[2] + this.rowD[1] * matrix.rowB[2] + this.rowD[2] * matrix.rowC[2] + this.rowD[3] * matrix.rowD[2],
            this.rowD[0] * matrix.rowA[3] + this.rowD[1] * matrix.rowB[3] + this.rowD[2] * matrix.rowC[3] + this.rowD[3] * matrix.rowD[3]
        ];
        return new Matrix(result);
    }
}
