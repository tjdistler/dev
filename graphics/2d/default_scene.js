
var scene_def = {
  fps: 30,
  duration: 3 * 1000, // ms
  objects: [ // list of geometry objects and their transforms.
    { // object0
      triangles: [ // triangles defining the object.
        [
          [-1.0,-0.5],
          [ 1.0,-0.5],
          [ 0.0, 0.5]
        ],
        [
          [ 1.0,-0.5],
          [-1.0,-0.5],
          [ 0.0,-1.5]
        ]
      ],
      position: [0, 0], // in the world
      transforms: [
        { // transform0 - order matters
          begin_ts: 0.25 * 1000,
          end_ts: 1.5 * 1000,
          type: 'rotate',
          angle: 180
        },
        { // transform0 - order matters
          begin_ts: 1.5 * 1000,
          end_ts: 2.75 * 1000,
          type: 'rotate',
          angle: -180
        },
        { // transform1
          begin_ts: 0, // ms
          end_ts: 1.5 * 1000, // ms
          type: 'translate',
          dx: 7,
          dy: 1
        },
        { // transform2
          begin_ts: 1.5 * 1000, // ms
          end_ts: 3 * 1000, // ms
          type: 'translate',
          dx: -7,
          dy: -1
        }
      ]
    },
    { // object1
      triangles: [
        [
          [-1.0, 0.0],
          [-0.5,-1.0],
          [ 0.0, 0.0]
        ]
      ],
      position: [-1.25, -1.0],
      transforms: [
        {
          begin_ts: 1 * 1000,
          end_ts: 2.5 * 1000,
          type: 'scale',
          dx: 2,
          dy: 3
        },
        {
          begin_ts: 0.5 * 1000,
          end_ts: 2 * 1000,
          type: 'translate',
          dx: -1,
          dy: 4
        },
        {
          begin_ts: 0.25 * 1000,
          end_ts: 2 * 1000,
          type: 'rotate',
          angle: 180
        }
      ]
    },
    { // object2
      triangles: [
        [
          [-1.0,-1.0],
          [ 1.0,-1.0],
          [ 0.0, 1.0]
        ]
      ],
      position: [-3.0, 3.0],
      begin_ts: 1 * 1000, //ms when object appears
      end_ts: 3*1000,
      transforms: [
        {
          type: 'rotate',
          angle: 360
        },
        {
          begin_ts: 0,
          end_ts: 1.5 * 1000,
          type: 'scale',
          dx: 2,
          dy: 2
        },
        {
          begin_ts: 1.5 * 1000,
          end_ts: 3 * 1000,
          type: 'scale',
          dx: 0.5,
          dy: 0.5
        }
      ]
    }
  ]
}
