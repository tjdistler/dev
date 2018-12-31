// Main application javascript file.
"use strict";

var CELSIUS_SCALAR = 10000;

function enableRefreshBtn() {
    $('#refreshBtn').prop('disabled',false);
    $('#refreshSpinner').hide();
}


// ON LOAD
$(function() {
	console.log('ready');
	
    // Test for local storage support
    if (typeof(Storage) === 'undefined') {
        alert('Local storage not supported!');
    }
    else {
        $('#inDeviceID').val( localStorage.deviceID );
        $('#inAuthToken').val( localStorage.authToken );
    }
    
    onRefresh(false);
});


// alertUser = true mean show alert box on error.
function onRefresh(alertUser) {
    console.log('onRefresh');
    
    $('#refreshBtn').prop('disabled',true);
    $('#refreshSpinner').show();
    
    var deviceID  = $('#inDeviceID').val();
    var authToken = $('#inAuthToken').val();
    
    if (deviceID.length === 0)
        deviceID = localStorage.deviceID;
    if (authToken.length === 0)
        authToken = localStorage.authToken;
    
    if (deviceID === undefined || authToken === undefined) {
        if (alertUser)
            alert('Device ID and authorization token must be defined!');
        enableRefreshBtn();
        return;
    }
    
    localStorage.deviceID = deviceID;
    localStorage.authToken = authToken;
    
    apiGetState(authToken, deviceID, function(err, state) {
        
        if (err != undefined) {
            if (alertUser)
                alert('Refresh state failed! ' + err);
            enableRefreshBtn();
            return;
        }
        
        console.log('State: ' + JSON.stringify(state));
        
        
        apiGetSettings(authToken, deviceID, function(err, settings) {
            
            if (err != undefined) {
                if (alertUser)
                    alert('Refresh settings failed! ' + err);
                enableRefreshBtn();
                return;
            }
            
            console.log('Settings: ' + JSON.stringify(settings));
            
            var temp0 = state.t[0];
            var temp1 = state.t[1];
            var ctrlState = toStateString( state.s );

            if (temp0 != '--')
                temp0 = CtoF(parseInt(temp0)) + '&deg;';
            if (temp1 != '--')
                temp1 = CtoF(parseInt(temp1)) + '&deg;';
            
            var setpoint = CtoF(settings.sp);
            var tolerance = ( CtoF(settings.tl) - 32.0 ).toFixed(1);
            var asc = MStoM(settings.asc);
            var offset0 = ( CtoF(settings.o[0]) - 32.0 ).toFixed(1);
            var offset1 = ( CtoF(settings.o[1]) - 32.0 ).toFixed(1);
            var ledLevel = settings.led;
            var Kp = settings.kp;
            var Ki = settings.ki;
            var Kd = settings.kd;

            var autoSetpoint = CtoF(settings.asp);
            var autoTimePeriod = StoH(settings.atp);

            $('#temp0').html(temp0);
            $('#temp1').html(temp1);
            $('#state').html(ctrlState);
            
            $('#inSetpoint').val(setpoint);
            $('#inTolerance').val(tolerance);
            $('#inASC').val(asc);
            $('#inOffset0').val(offset0);
            $('#inOffset1').val(offset1);

            updateLEDLevelSlider(ledLevel);
            
            $('#inHeatEnabled').prop('checked', settings.h == 1);
            $('#inCoolEnabled').prop('checked', settings.c == 1);
            
            $('#inPidKp').val(Kp);
            $('#inPidKi').val(Ki);
            $('#inPidKd').val(Kd);

            $('#inAutoAdjustEnabled').prop('checked', settings.aa == 1);
            $('#inAutoSetpoint').val(autoSetpoint);
            $('#inAutoTimePeriod').val(autoTimePeriod);

            enableRefreshBtn();
        });
    });
}


function onApply() {
    console.log('onApply');

    var deviceID    = $('#inDeviceID').val();
    var authToken   = $('#inAuthToken').val();
    
    var setpoint    = FtoC( $('#inSetpoint').val() );
    var tolerance   = FtoC( $('#inTolerance').val() ) + 177778;
    var asc         = MtoMS( $('#inASC').val() );
    var offset0     = FtoC( $('#inOffset0').val() ) + 177778;
    var offset1     = FtoC( $('#inOffset1').val() ) + 177778;
    var ledLevel    = $('#inLEDLevel').val();
    var heatEnabled = $('#inHeatEnabled').is(':checked');
    var coolEnabled = $('#inCoolEnabled').is(':checked');
    var Kp          = $('#inPidKp').val();
    var Ki          = $('#inPidKi').val();
    var Kd          = $('#inPidKd').val();

    var autoAdjustEnabled = $('#inAutoAdjustEnabled').is(':checked');
    var autoSetpoint      = FtoC( $('#inAutoSetpoint').val() );
    var autoTimePeriod    = HtoS( $('#inAutoTimePeriod').val() );
    
    if (tolerance < 0.5 * CELSIUS_SCALAR) {
        alert('Tolerance value too low!');
        return;
    }
    if (autoTimePeriod < 1) {
        alert('Auto-adjust time period is too low!');
        return;
    }
    
    localStorage.deviceID = deviceID;
    localStorage.authToken = authToken;
    
    console.log('Values: ', deviceID, authToken, setpoint, tolerance, asc, offset0, offset1, ledLevel, heatEnabled,
                coolEnabled, Kp, Ki, Kd, autoAdjustEnabled, autoSetpoint, autoTimePeriod);
    
    apiApplySettings(authToken, deviceID, setpoint, tolerance, asc, offset0, offset1, 
                     ledLevel, heatEnabled, coolEnabled, Kp, Ki, Kd, autoAdjustEnabled,
                     autoSetpoint, autoTimePeriod, function(err) {
        
        if (err) {
            alert('Apply settings failed! ' + err);
            return;
        }
    });
}


function onLEDLevelChange() {
    var level = $('#inLEDLevel').val();
    $('#ledPercent').html(level + "%");
}

function updateLEDLevelSlider(level) {
    $('#inLEDLevel').val(level);
    $('#ledPercent').html(level + "%");
} 


function FtoC(F) {
    return Math.round( ((F - 32.0) / 1.8) * CELSIUS_SCALAR );
}

function CtoF(C) {
    return parseFloat( ((C / CELSIUS_SCALAR) * 1.8 + 32.0).toFixed(1) );
}

function MtoMS(M) {
    return (M * 60.0 * 1000.0).toFixed(0);
}

function MStoM(MS) {
    return (MS / 1000.0 / 60.0).toFixed(0);
}

function HtoS(H) {
    return (H * 3600).toFixed(0);
}

function StoH(S) {
    return (S / 3600.0).toFixed(0);
}

function toStateString(state) {
    if (state === 0)
        return 'Off';
    if (state === 1)
        return 'Heat ASC';
    if (state === 2)
        return 'Heating';
    if (state === -1)
        return 'Cool ASC';
    if (state === -2)
        return 'Cooling';
    return '<unknown>';
}
