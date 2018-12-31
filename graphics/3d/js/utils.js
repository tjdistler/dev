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