// Main application javascript file.
"use strict";

var gChartOptions = {
    tooltips: {
        enabled: true
    }
};


$(function() {
	console.log('ready');
    
    // Enable tooltips
    $('[data-toggle="popover"]').popover();
	
});


function runSimulation() {
    console.log('Running simulation...');
    
    // Gather user inputs.
    var beerVolumeL            = parseFloat($('#beerVolume').val());
    var beerStartTempC         = parseFloat($('#beerStartTemp').val());
    var externalStartTempC     = parseFloat($('#externalStartTemp').val());
    var targetTempC            = parseFloat($('#targetTemp').val());
    var toleranceC             = parseFloat($('#tolerance').val());
    var ascM                   = parseFloat($('#asc').val());
    var runtimeM               = parseFloat($('#runtime').val());
    var stepTimeS              = parseFloat($('#stepTime').val());
    var chillerVolumeM3        = parseFloat($('#chillerVolume').val());
    var chillerOutputW         = parseFloat($('#chillerOutput').val());
    var heaterOutputW          = parseFloat($('#heaterOutput').val());
    var heaterRiseTimeS        = parseFloat($('#heaterRiseTime').val());
    var heaterContactP         = parseFloat($('#heaterContact').val()) / 100.0;
    var fermentationHeatW      = parseFloat($('#fermentationHeat').val());
    var thermalConductivityWmC = parseFloat($('#thermalConductivity').val());
    var surfaceAreaM2          = parseFloat($('#surfaceArea').val());
    var surfaceThicknessM      = parseFloat($('#surfaceThickness').val());
    var surfaceExposedP        = parseFloat($('#surfaceExposed').val()) / 100.0;
    
    console.log('Simulation parameters: ', 
        beerVolumeL,
        beerStartTempC,
        externalStartTempC,
        targetTempC,
        toleranceC,
        ascM,
        runtimeM,
        stepTimeS,
        chillerVolumeM3,
        chillerOutputW,
        heaterOutputW,
        heaterRiseTimeS,
        heaterContactP,
        fermentationHeatW,
        thermalConductivityWmC,
        surfaceAreaM2,
        surfaceThicknessM,
        surfaceExposedP
    );
    
    if (heaterContactP < 0 || heaterContactP > 1) {
        alert('Invalid heater contact percent value!');
        return;
    }
    
    if (surfaceExposedP < 0 || surfaceExposedP > 1) {
        alert('Invalid fermenter exposed percent value!');
        return;
    }
    
    var simulator = new Simulator(
        beerVolumeL,
        beerStartTempC,
        externalStartTempC,
        targetTempC,
        toleranceC,
        ascM,
        runtimeM,
        stepTimeS,
        chillerVolumeM3,
        chillerOutputW,
        heaterOutputW,
        heaterRiseTimeS,
        heaterContactP,
        fermentationHeatW,
        thermalConductivityWmC,
        surfaceAreaM2,
        surfaceThicknessM,
        surfaceExposedP
    );
    
    $('#temperatureChartDiv').html('');
    $('#pidOutputChartDiv').html('');
    $('#heaterOutputChartDiv').html('');
    $('#chamberTempChartDiv').html('');
    
    setTimeout(function() {
        $('#temperatureChartDiv').html('<canvas id="temperatureChart" width="400" height="200"></canvas>');
        $('#pidOutputChartDiv').html('<canvas id="pidOutputChart" width="400" height="200"></canvas>');
        $('#heaterOutputChartDiv').html('<canvas id="heaterOutputChart" width="400" height="200"></canvas>');
        $('#chamberTempChartDiv').html('<canvas id="chamberTempChart" width="400" height="200"></canvas>');
        $('#progress').css('width', '0%').html('0%');
        
        simulator.run(function() {
            chartResults(simulator);
        });
    }, 1);
}


function chartResults(simulator) {
    console.log('Charting results...');    
    
    drawTemperatureChart( simulator.getBeerTempData(), simulator.getTargetTemp(), simulator.getTolerance() );
    drawPidOutputChart( simulator.getPidOutputData() );
    drawHeaterOutputChart( simulator.getHeaterOutputData() );
    drawChamberTempChart( simulator.getChillerOutputData() );
    
    console.log('Complete!');
}


function drawTemperatureChart(beerTempData, targetTemp, tolerance) {
    var chartData = {
        labels: [],
        datasets: [
            {
                label: 'Beer temperature (C)',
                fill: false,
                backgroundColor: 'blue',
                borderColor: 'blue',
                data: beerTempData
            },
            {
                label: 'Setpoint (C)',
                fill: false,
                backgroundColor: 'green',
                borderColor: 'green',
                data: []
            },
            {
                label: 'Tolerance High',
                fill: false,
                backgroundColor: 'lightgray',
                borderColor: 'lightgray',
                data: []
            },
            {
                label: 'Tolerance Low',
                fill: false,
                backgroundColor: 'lightgray',
                borderColor: 'lightgray',
                data: []
            },
            {
                label: 'Average temperature (C)',
                fill: false,
                backgroundColor: 'orange',
                borderColor: 'orange',
                data: []
            }
        ]
    };
    
    // Set setpoint data
    for (var ii=0; ii<beerTempData.length; ++ii)
        chartData.datasets[1].data.push(targetTemp);
    
    // Set tolerance high data
    for (var ii=0; ii<beerTempData.length; ++ii)
        chartData.datasets[2].data.push(targetTemp + tolerance);
    
    // Set tolerance low data
    for (var ii=0; ii<beerTempData.length; ++ii)
        chartData.datasets[3].data.push(targetTemp - tolerance);
    
    // Set average temperature
    for (var ii=0; ii<beerTempData.length; ++ii) {
        var sum = 0;
        for (var jj=0; jj<=ii; ++jj)
            sum += beerTempData[jj];
        chartData.datasets[4].data.push( sum / (ii+1) );
    }
    
    // Set labels
    for (var ii=0; ii<beerTempData.length; ++ii)
        chartData.labels.push(ii);
    
    var ctx = $('#temperatureChart');
    var temperatureChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: gChartOptions
    });
}


function drawPidOutputChart(data) {
    var chartData = {
        labels: [],
        datasets: [
            {
                label: 'PID Output',
                fill: false,
                backgroundColor: 'green',
                borderColor: 'green',
                data: data
            }
        ]
    };
    
    for (var ii=0; ii<data.length; ++ii)
        chartData.labels.push(ii);
    
    var ctx = $('#pidOutputChart');
    var heaterOutputChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: gChartOptions
    });
}


function drawHeaterOutputChart(data) {
    var chartData = {
        labels: [],
        datasets: [
            {
                label: 'Heater output (W)',
                fill: false,
                backgroundColor: 'red',
                borderColor: 'red',
                data: data
            }
        ]
    };
    
    for (var ii=0; ii<data.length; ++ii)
        chartData.labels.push(ii);
    
    var ctx = $('#heaterOutputChart');
    var heaterOutputChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: gChartOptions
    });
}


function drawChamberTempChart(data) {
    var chartData = {
        labels: [],
        datasets: [
            {
                label: 'Fermentation chamber temperature (C)',
                fill: false,
                backgroundColor: 'blue',
                borderColor: 'blue',
                data: data
            }
        ]
    };
    
    for (var ii=0; ii<data.length; ++ii)
        chartData.labels.push(ii);
    
    var ctx = $('#chamberTempChart');
    var heaterOutputChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: gChartOptions
    });
}



// Used to have a form do nothing on submit.
function formDoNothing() {
}