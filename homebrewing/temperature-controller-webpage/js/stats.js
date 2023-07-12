// Main application javascript file.
"use strict";

var dataBtnTypes = [
    {label:"30m", mins:30,           scale:0},
    {label:"60m", mins:60,           scale:0},
    {label:"4h",  mins:4 * 60,       scale:0},
    {label:"8h",  mins:8 * 60,       scale:0},
    {label:"16h", mins:16 * 60,      scale:0},
    {label:"24h", mins:24 * 60,      scale:10},
    {label:"48h", mins:48 * 60,      scale:30},
    {label:"72h", mins:72 * 60,      scale:30},
    {label:"5d",  mins:5 * 24 * 60,  scale:60},
    {label:"10d", mins:10 * 24 * 60, scale:60}
];

var TS_SETPOINT_FIELD_ID = 1;
var TS_TEMPERATURE_FIELD_ID = 2;
var TS_OUTPUT_STATE_FIELD_ID = 3;
var TS_PID_OUTPUT_FIELD_ID = 4;
var TS_TOLERANCE_FIELD_ID = 5;
var TS_AVERAGE_TEMP_FIELD_ID = 6;


// Plotly chart layout
var chartLayout = {
    displayModeBar: false,
    legend: {
        "orientation": "h",
        x: 0,
        y: 110
    },
    showlegend: true,
    autosize: false,
    width: window.innerWidth,  // Set programmatically later.
    height: 275,
    margin: {
        l: 40,
        r: 5,
        b: 95,
        t: 0,
        pad: 4
    }
};

var CHART_WIDTH_ADJUST = -10;
var isMobile = window.matchMedia("only screen and (max-width: 760px)");


// ON LOAD
$(function() {
	console.log('ready');
    
    // Set default chart width
    chartLayout.width = window.innerWidth + CHART_WIDTH_ADJUST;  
    $(window).resize(onWindowResize);
	
    // Test for local storage support
    if (typeof(Storage) === 'undefined') {
        alert('Local storage not supported!');
    }
    else {
        console.log( localStorage.deviceID );
        console.log( localStorage.authToken );
    }
    
    // Populate buttons
    var btnGroup = $('#time-btn-group');
    for (var ii=0; ii<dataBtnTypes.length; ++ii) {
        btnGroup.append('<button type="button" class="btn btn-default data-time-btn" onclick="onStatsTimeBtnClicked(' + ii + 
                        '); return false;">' + dataBtnTypes[ii].label + '</button>');
    }    
});


function onWindowResize() {
    var newWidth = window.innerWidth + CHART_WIDTH_ADJUST;
    
    chartLayout.width = newWidth;
    
    var update = {
        width: newWidth,
        height: chartLayout.height
    };

    Plotly.relayout('chart1', update);
    Plotly.relayout('chart2', update);
    Plotly.relayout('chart3', update);
}


function onStatsTimeBtnClicked(index) {
    
    // Clear existing graphs
    $('#chart1Div').html('<div id="chart1"></div><div id="chart2"></div><div id="chart3"></div>');
    $('#status').append('<p>Retrieving data...</p>');
                         
    var duration = dataBtnTypes[index].mins;
    var scale = dataBtnTypes[index].scale;
    
    var endTime = moment();
    var startTime = moment();
    startTime.subtract(duration, 'minutes');
    var format = 'YYYY-MM-DD HH:mm:ss';

    var totalData = [];
    
    // start = start time as ISO8601 string
    // end = start time as ISO8601 string
    // cb = function(error, data)
    var getMore = function(start, end, cb) {
        
        $('#status').append('<p>' + start + ' - ' + end + '</p>');
        
        tsApiGetChannel(start, end, scale, function(err, channel) {
            if (err) {
                return cb(err);
            }

            console.log('Processing received data. Size: ' + channel.feeds.length);
            $('#status').append('<p>&nbsp;&nbsp;Entries: ' + channel.feeds.length + '</p>');

            cb(null, channel.feeds);
        });
    };
    
    // Keep making calls until all data is received.
    var handler = function(err, data) {
        if (err) {
            alert('Failed to get all the requested data! ' + err);
            return;
        }

        // Keep getting data until no more is returned.
        if (data.length == 0)
            return _processAndGraph(totalData);
        
        totalData = data.concat(totalData);
        
        // Make sure we got it all... look at the oldest time returned and see if it's newer than or start time.
        var oldest = moment(data[0].created_at);
        if (startTime >= oldest)
            return _processAndGraph(totalData);

        getMore(startTime.format(format), oldest.subtract(1, 'seconds').format(format), handler);
    };
    
    // Kick off the process
    getMore(startTime.format(format), endTime.format(format), handler);
}


function _processAndGraph(data) {

    try {
        $('#status').html('');
        
        // Extract values
        var timestamps = [];
        var setpoints = [];
        var temperatures = [];
        var averageTemps = [];
        var outputStates = [];
        var pidOutputs = [];
        var tolerancesHigh = [];
        var tolerancesLow = [];
        data.forEach(function(value, index) {
            timestamps.push( moment(value.created_at).format('M/D HH:mm') );
            setpoints.push(parseFloat( value['field' + TS_SETPOINT_FIELD_ID] ));
            temperatures.push(parseFloat( value['field' + TS_TEMPERATURE_FIELD_ID] ));
            averageTemps.push(parseFloat( value['field' + TS_AVERAGE_TEMP_FIELD_ID] ));
            outputStates.push(parseFloat( value['field' + TS_OUTPUT_STATE_FIELD_ID] ));
            pidOutputs.push(parseFloat( value['field' + TS_PID_OUTPUT_FIELD_ID] ));
            tolerancesHigh.push( setpoints[index] + parseFloat( value['field' + TS_TOLERANCE_FIELD_ID] ));
            tolerancesLow.push( setpoints[index] - parseFloat( value['field' + TS_TOLERANCE_FIELD_ID] ));
        });

        // Calculate stats
        var stat = jStat(temperatures);        
        var latest = temperatures[temperatures.length-1].toFixed(2);
        var expAvg = averageTemps[averageTemps.length-1].toFixed(2);
        var average = stat.mean().toFixed(2);
        var min = stat.min().toFixed(2);
        var max = stat.max().toFixed(2);
        var range = stat.range().toFixed(2);
        var stdev = stat.stdev().toFixed(3);
        var variance = stat.variance().toFixed(3);

        $('#statCurrent').html(latest + '&deg;');
        $('#statExpAverage').html(expAvg + '&deg;');
        $('#statMin').html(min + '&deg;');
        $('#statMax').html(max + '&deg;');
        $('#statRange').html(range + '&deg');
        $('#statAverage').html(average + '&deg;');
        $('#statStdev').html(stdev + '&deg;');
        $('#statVariance').html(variance + '&deg;');

        // Calculate average
        var sum = 0;
        var averages = [];
        temperatures.forEach(function(value, index) {
            sum += value;
            averages.push( sum / (index+1));
        });

        // Plot
        var temperature = {
            x: timestamps,
            y: temperatures,
            mode: 'lines',
            name: 'Temp (&deg;F)',
            line: { color: '#0275d8', dash: 'solid', width: 2 }
        };
        
        var averageTemp = {
            x: timestamps,
            y: averageTemps,
            mode: 'lines',
            name: 'EMA (&deg;F)',
            line: { color: 'orange', dash: 'dot', width: 2 }
        };

       /* var average = {
            x: timestamps,
            y: averages,
            mode: 'lines',
            name: 'Average',
            line: { color: 'orange', dash: 'dot', width: 2 }
        }; */
        
        var setpoint = {
            x: timestamps,
            y: setpoints,
            mode: 'lines',
            name: 'Setpoint',
            showlegend: false,
            line: { color: 'black', dash: 'solid', width: 1 }
        };
        
        var toleranceHigh = {
            x: timestamps,
            y: tolerancesHigh,
            mode: 'lines',
            name: 'Tolerance',
            showlegend: false,
            line: { color: 'grey', dash: 'dot', width: 1 }
        };
        
        var toleranceLow = {
            x: timestamps,
            y: tolerancesLow,
            mode: 'lines',
            name: 'Tolerance',
            showlegend: false,
            line: { color: 'grey', dash: 'dot', width: 1 }
        };

        var outputState = {
            y: outputStates,
            mode: 'lines',
            name: 'Output State',
            line: { color: 'red', dash: 'solid', width: 2 }
        };
        
        var pidOutput = {
            y: pidOutputs,
            mode: 'lines',
            name: 'PID Output',
            line: { color: 'green', dash: 'solid', width: 2 }
        };

        Plotly.newPlot('chart1', [setpoint, averageTemp, toleranceHigh, toleranceLow, /*average,*/ temperature], chartLayout);

        var smallerLayout = jQuery.extend(true, {}, chartLayout);
        smallerLayout.height = 210;
        smallerLayout.margin.b = 30;
        
        Plotly.newPlot('chart2', [outputState], smallerLayout);
        
        Plotly.newPlot('chart3', [pidOutput], smallerLayout);
    }
    catch (e) {
        alert('An error occurred processing the received data! ' + e);
    }
}