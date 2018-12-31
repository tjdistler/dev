"use strict"

var BASE_URL = 'https://api.particle.io/v1/devices';
var THINKSPEAK_BASE_URL = 'https://api.thingspeak.com/channels/210613'


//cb = function(error, jsonResult)
//
/* result example:
{
    "channel": {
        "id":210613,
        "name":"PID Channel",
        "latitude":"0.0",
        "longitude":"0.0",
        "field1":"Set Point",
        "field2":"Temperature",
        "field3":"Output State",
        "field4":"PID Output",
        "field5":"Tolerance",
        "created_at":"2017-01-06T00:36:40Z",
        "updated_at":"2017-02-03T18:56:33Z",
        "last_entry_id":112970
    },
    "feeds": [
        {"created_at":"2017-02-03T11:26:47Z","entry_id":111396,"field2":"75.98750"},
        ...
    ]
}
*/
function tsApiGetField(fieldNum, startTime, cb) {
    try {
        var url = THINKSPEAK_BASE_URL + '/fields/' + fieldNum + '.json?timezone=America/Denver';
        if (startTime !== undefined)
            url += '&start=' + startTime;
        
        $.ajax({
            url: url,
            method: 'GET',
            context: this,
            useDefaultXhrHeader: false,
            headers: {
                'Accept' : 'application/json'
            },
            jsonp: false
        })
        .done(function(result, textStatus) {
            return cb( null, result );
        })
        .fail(function(jqXHR) {
            return cb(jqXHR.responseText);
        });
    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}


function tsApiGetChannel(startTime, endTime, scale, cb) {
    try {
        var url = THINKSPEAK_BASE_URL + '/feeds.json?timezone=America/Denver';
        if (startTime !== null)
            url += '&start=' + startTime;
        if (endTime !== null)
            url += '&end=' + endTime;
        if (scale !== null && scale > 0)
            url += '&timescale=' + scale;
        
        $.ajax({
            url: url,
            method: 'GET',
            context: this,
            useDefaultXhrHeader: false,
            headers: {
                'Accept' : 'application/json'
            },
            jsonp: false
        })
        .done(function(result, textStatus) {
            return cb( null, result );
        })
        .fail(function(jqXHR) {
            return cb(jqXHR.responseText);
        });
    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}


//cb = function(error, jsonResult)
function apiGetState(authToken, deviceID, cb) {
    try {
        $.ajax({
            url: BASE_URL + '/' + deviceID + '/state',
            method: 'GET',
            context: this,
            useDefaultXhrHeader: false,
            headers: {
                'Authorization': 'Bearer ' + authToken,
                'Accept' : 'application/json'
            },
            jsonp: false
        })
        .done(function(result, textStatus) {
            return cb( null, JSON.parse(result.result) );
        })
        .fail(function(jqXHR) {
            return cb(jqXHR.responseText);
        });
    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}




//cb = function(error, jsonResult)
function apiGetSettings(authToken, deviceID, cb) {
    try {
        $.ajax({
            url: BASE_URL + '/' + deviceID + '/settings',
            method: 'GET',
            context: this,
            useDefaultXhrHeader: false,
            headers: {
                'Authorization': 'Bearer ' + authToken,
                'Accept' : 'application/json'
            },
            jsonp: false
        })
        .done(function(result, textStatus) {
            return cb( null, JSON.parse(result.result) );
        })
        .fail(function(jqXHR) {
            return cb(jqXHR.responseText);
        });
    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}



//cb = function(error, text)
function apiGetLastError(authToken, deviceID, cb) {
    try {
        $.ajax({
            url: BASE_URL + '/' + deviceID + '/error',
            method: 'GET',
            context: this,
            useDefaultXhrHeader: false,
            headers: {
                'Authorization': 'Bearer ' + authToken,
                'Accept' : 'application/json'
            },
            jsonp: false
        })
        .done(function(result, textStatus) {
            return cb( null, result.result );
        })
        .fail(function(jqXHR) {
            return cb(jqXHR.responseText);
        });
    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}


//cb = function(error)
function apiApplySettings(authToken, deviceID, setpoint, tolerance, asc, offset0, offset1, 
                          ledLevel, heatEnabled, coolEnabled, Kp, Ki, Kd, cb) {
    try {
        // Limit the data length to conform to cloud limitations..
        var data1 = '{sp:' + setpoint + ',tl:' + tolerance + ',asc:' + asc + 
            ',o:[' + offset0 + ',' + offset1 + '],led:' + ledLevel +  '}';
        
        var data2 = '{h:' + (heatEnabled?1:0) + ',c:' + (coolEnabled?1:0) +
            ',kp:' + Kp + ',ki:' + Ki + ',kd:' + Kd + '}';
        
        _apiApplySettings(authToken, deviceID, data1, function(err) {
            if (err)
                return cb(err);
            _apiApplySettings(authToken, deviceID, data2, cb);
        })

    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}



//cb = function(error)
function _apiApplySettings(authToken, deviceID, data, cb) {
    try {
        $.ajax({
            url: BASE_URL + '/' + deviceID + '/settings',
            method: 'POST',
            context: this,
            useDefaultXhrHeader: false,
            headers: {
                'Authorization': 'Bearer ' + authToken
            },
            data: {
                args: data
            }
        })
        .done(function(result, textStatus) {
            
            if (result.return_value !== undefined && result.return_value == -1) {
                apiGetLastError(authToken, deviceID, function(err, text) {
                    if (text.length == 0)
                        text = 'Unknown API error!';
                    return cb(text);
                })
            }
            else
                return cb();
        })
        .fail(function(jqXHR) {
            return cb(jqXHR.responseText);
        });
    }
    catch (e) {
        return cb(e.toLocaleString());
    }
}