// MAIN

// Run unit tests.
run_unittests();

var scene = scene_def;

// Setup JSON editor
var editContainer = document.getElementById('jsoneditor');
var editOptions = {
    onEditable: onJsonEdit,
    mode: 'text'
};
var editor = new JSONEditor(editContainer, editOptions);

editor.set(scene);

viewport = new Viewport('viewport');

var animator = new Animator(viewport, scene);
animator.loop = document.getElementById('loop_checkbox').checked;
animator.hud = document.getElementById('hud_checkbox').checked;
animator.play();


function onJsonEdit(node)
{
    // Don't allow any edits
    return false;
}

function onLoopCheckboxClick()
{
    if (document.getElementById('loop_checkbox').checked)
    {
        animator.stop();
        animator.loop = true;
        animator.play();
    }
    else
        animator.loop = false;
}


function onHudCheckboxClick()
{
    animator.hud = document.getElementById('hud_checkbox').checked
}