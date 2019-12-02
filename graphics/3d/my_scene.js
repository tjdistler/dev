
var scene_def = {
    fps: 30,
    duration: 0.1,//4 * 1000, // ms
    width: 320,
    height: 240,
    camera: {
        position: [0, 0, 1],
        look_at: [0, 0, 0],
        fov: 90,
        transforms: []
    },
    objects: [ // list of geometry objects and their transforms.
        { // object0
            triangles: [ // triangles defining the object.
                [
                    { // Vertex
                        pos: [-0.5, -0.2887, 0.0],
                        color: [1, 0, 0, 1]
                    },
                    {
                        pos: [0.5, -0.2887, 0.0],
                        color: [0, 1, 0, 1]
                    },
                    {
                        pos: [0.0,  0.5774, 0.0],
                        color: [0, 0, 1, 1]
                    }
                ]
            ],
            position: [0, 0, 0], // in the world
            transforms: [
            /*    { // transform0 - order matters
                    type: 'rotate',
                    dz: 360
                },*/
                {
                    type: 'scale',
                    end_ts: 0,
                    dx: 4,
                    dy: 4
                }
            ]
        }
    ]
}
