// Defines a vertex

function Vertex() {}

/*
 * position - Vector specifying the vertex location.
 * color - Vector specifying the vertex color.
 * normal - Optional: Vector specifying the vertex normal.
 */
function Vertex(position, color, normal)
{
    this.pos = position; // Vector
    this.color = color;
    this.normal = normal;
}

Vertex.prototype.setPosition = function(position) {
    this.pos = position;
}

Vertex.prototype.setColor = function(color) {
    this.color = color;
}

Vertex.prototype.setNormal = function(normal) {
    this.normal = normal;
}

// Set a function used to calculate the normal on-demand.
Vertex.prototype.setNormalFunc = function(normalFunc) {
    this.normalFunc = normalFunc;
}

Vertex.prototype.getPosition = function() {
    return this.pos;
}

Vertex.prototype.getColor = function() {
    return this.color;
}

Vertex.prototype.getNormal = function() {
    if (!this.normal)
        this.normal = this.normalFunc();
    return this.normal;
}