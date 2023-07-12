// Temperature controller implementation
"use strict";


var Controller = function (targetTempC, toleranceC, ascM, stepTimeS) {
    console.log('Controller created');
    
    this._targetTemp = targetTempC;
    this._toleranceC = toleranceC;
    this._ascMs = ascM * 60 * 1000;
    this._stepTime = stepTimeS;
    this._heatOn = false;
    this._coolOn = false;
    
    this._currentStep = 0;
    this._errorSum = 0;
    this._iterationOfLastOff = -1; // step when output last went off (for ASC)
    
    this._pidOutputData = [];
};

Controller.prototype.heatOn = function() {
    return this._heatOn;
}

Controller.prototype.coolOn = function() {
    return this._coolOn;
}

Controller.prototype.getPidOutputData = function() {
    return this._pidOutputData;
}


// Called for each simulation step.
Controller.prototype.onStep = function(probeTemperatureC) {
    
    // Slowly phase-out old error to account for a large error at controller power-on.
    var error = probeTemperatureC - this._targetTemp;
    this._errorSum = (0.1 * error) + (0.98 * this._errorSum);
    
    if (this._errorSum > this._toleranceC)
        this._errorSum = this._toleranceC;
    else if (this._errorSum < -this._toleranceC)
        this._errorSum = -this._toleranceC;
    
    var adjustedTargetTemp = this._targetTemp - this._errorSum;
    
    if (this._heatOn) {
        if (probeTemperatureC >= adjustedTargetTemp) {
            this._heatOn = false;
            this._iterationOfLastOff = this._currentStep++;
            return;
        }
    }
    
    if (this._coolOn) {
        if (probeTemperatureC <= adjustedTargetTemp) {
            this._coolOn = false;
            this._iterationOfLastOff = this._currentStep++;
            return;
        }
    }
    
    // Anti-short cycle
    var delta = 0;
    if (this._iterationOfLastOff != -1)
        delta = (this._currentStep - this._iterationOfLastOff) * this._stepTime * 1000;
    
    if (this._iterationOfLastOff == -1 || delta >= this._ascMs) {
        if (probeTemperatureC < adjustedTargetTemp - this._toleranceC) {
            this._heatOn = true;
        }
        else if (probeTemperatureC > adjustedTargetTemp + this._toleranceC) {
            this._coolOn = true;
        }
    }
    
    ++this._currentStep;
    this._pidOutputData.push(this._errorSum);
};