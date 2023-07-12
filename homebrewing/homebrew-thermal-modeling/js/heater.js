// Models the heater object
"use strict";


var Heater = function (outputW, riseTimeS, stepTimeS) {
    this._outputW = outputW;
    this._riseTimeS = riseTimeS;
    this._stepTimeS = stepTimeS;
    
    this._on = false;
    this._lastStateChangeStep = null; // flags that this value has never been set.
    
    this._currentOutputW = 0;
    this._currentStep = 0;
};


// Watts
Heater.prototype.getOutputPower = function() {
    return this._currentOutputW;
}


Heater.prototype.updateState = function(heatOn) {
    
    if (this._on != heatOn) {
        if ( (heatOn && this.getScalar() == 0) ||
             (!heatOn && this.getScalar() == 1) )
        {
            this._lastStateChangeStep = this._currentStep;
        }
    }
    
    this._on = heatOn;

    this._currentOutputW = this.getScalar() * this._outputW * this._stepTimeS;
    
    ++this._currentStep;
}


Heater.prototype.getScalar = function() {
    return this._on ? this._getScalarOn() : this._getScalarOff();
}


Heater.prototype._getScalarOn = function() {
    
    if (this._lastStateChangeStep == undefined) {
        return 1;
    }
    
    var delta = (this._currentStep - this._lastStateChangeStep) * this._stepTimeS;
    if (delta >= this._riseTimeS)
        return 1;
    
    var x = delta / this._riseTimeS;
    return x*x; // x^2
}


Heater.prototype._getScalarOff = function() {
    
    if (this._lastStateChangeStep == undefined) {
        return 0;
    }
    
    var delta = (this._currentStep - this._lastStateChangeStep) * this._stepTimeS;
    if (delta >= this._riseTimeS)
        return 0;
    
    var x = 1 - (delta / this._riseTimeS);
    return x*x; // x^2
}