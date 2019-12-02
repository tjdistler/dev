function assert(condition, message)
{
  if (!condition)
    throw message + '\n' + (new Error).stack;
}


// Deep-copies an array or object.
function deepCopy(source)
{
  var result, key, value;
  result = Array.isArray(source) ? [] : {};
  for (key in source)
  {
      value = source[key];
      result[key] = (typeof value === 'object') ? deepCopy(value) : value;
  }
  return result;
}


// Round floating point number to specified decimal place.
function round(x, decimal)
{
    var f = Math.pow(10, decimal);
    return Math.round(x * f) / f;
}

// Log triangle vertices [Vertex,Vertex,Vertex]
function logTriangleVertices(prefix, vertices)
{
    var decimals = 1;
    console.info(prefix);
    
    vertices.forEach(function(vertex, index) {        
        var details = '\t[' + index + ']  ';
        if (vertex.pos)
            details += 'pos: [' 
                + round(vertex.pos.x, decimals) + ',' 
                + round(vertex.pos.y, decimals) + ',' 
                + round(vertex.pos.z, decimals) + ',' 
                + round(vertex.pos.w, decimals) + '],\t';
        if (vertex.color)
            details += 'color: [' 
                         + round(vertex.color.x, decimals) + ',' 
                         + round(vertex.color.y, decimals) + ',' 
                         + round(vertex.color.z, decimals) + ',' 
                         + round(vertex.color.w, decimals) + '],\t';
        if (vertex.normal)
            details += 'normal: [' 
                         + round(vertex.normal.x, decimals) + ',' 
                         + round(vertex.normal.y, decimals) + ',' 
                         + round(vertex.normal.z, decimals) + ',' 
                         + round(vertex.normal.w, decimals) + ']';
        console.info(details);
    });
}