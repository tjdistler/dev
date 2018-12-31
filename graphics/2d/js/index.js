// MAIN

var scene = scene_def;

// Attempt to read scene definition from local storage
if (Storage !== undefined) {
    if (localStorage.scene !== undefined)
        scene = JSON.parse(localStorage.scene);
    
    localStorage.scene = JSON.stringify(scene_def);
}
else
  console.error('LocalStorage not supported in browser.');

// Setup JSON editor
var editContainer = document.getElementById('jsoneditor');
var editOptions = {};
var editor = new JSONEditor(editContainer, editOptions);

editor.set(scene);

viewport = new Viewport('viewport');

var animator = new Animator(viewport, scene);
animator.play(document.getElementById('loop_checkbox').checked);


// Handles JSON update button click
function onJsonUpdateClick()
{
  animator.stop();
  animator = {};
  scene = editor.get();
  if (Storage !== undefined)
    localStorage.scene = JSON.stringify(scene);
  animator = new Animator(viewport, scene);
  animator.play(true);
}


function onLoopCheckboxClick()
{
  animator.loop = document.getElementById('loop_checkbox').checked;
}

function onZoomSliderChange()
{
  animator.GRID_SCALE = document.getElementById('zoom_slider').value;
  document.getElementById('zoom_label').innerHTML = 'Zoom (' + animator.GRID_SCALE +
      'px): ';
}