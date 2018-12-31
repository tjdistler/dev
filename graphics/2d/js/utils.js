function assert(condition, message)
{
  if (!condition)
    throw message || "Assertion failed";
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