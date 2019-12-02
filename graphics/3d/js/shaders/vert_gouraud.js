// Vertex shader for Gouraud shading
// vertexIn - Vertex object
// constants - RenderConstants object
// args - User-defined object
// Returns a Vertex object in normalized screen coordinates
function VertShaderGouraud(vertexIn, constants, args)
{
    var vertexOut = new Vertex();
    
    // Simply convert to view space
    vertexOut.pos = vertexIn.getPosition().transform(constants.getViewMatrix());
    vertexOut.color = vertexIn.getColor();
    vertexOut.normal = vertexIn.getNormal().transform(constants.getViewMatrix());
    
    return vertexOut;
}