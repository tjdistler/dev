// Models the chiller object (affects external temperature)
// The curve that defines 
"use strict";

var SPECIFIC_HEAT_AIR = 1005; // joule/gram
var MASS_AIR_25C = 1.19; // grams/liter
var LITERS_PER_CUBIC_METER = 1000;

function cubicMetersToLiters(volumeM3) {
    return volumeM3 * LITERS_PER_CUBIC_METER;
}


var Chiller = function (chillerVolumeM3, chillerOutputW, startingTempC, stepTimeS) {
    this._chillerVolumeL = cubicMetersToLiters( chillerVolumeM3 );
    this._chillerOutputW = chillerOutputW;
    this._currentTempC = startingTempC;
    this._stepTimeS = stepTimeS;
    
    this._on = false;
    this._currentStep = 0;
};

Chiller.prototype.getCurrentTemp = function() {
    return this._currentTempC;
}


Chiller.prototype.updateState = function(coolOn) {
    
    if (coolOn) {
        this.addEnergy( this._chillerOutputW * this._stepTimeS );
    }
}


// This method does NOT have a time component. It assumes any time scaling has occurred.
Chiller.prototype.addEnergy = function(energyW) {
    
    // T(C) = Q/cm
    var mass = this._chillerVolumeL * MASS_AIR_25C;
    this._currentTempC += energyW / (SPECIFIC_HEAT_AIR * mass);
}