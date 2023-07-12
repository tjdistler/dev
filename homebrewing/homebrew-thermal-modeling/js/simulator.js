// The main implementaion of the simulator
"use strict";


var Simulator = function (
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
) {
    console.log('Simulator created');
    
    this._beerVolumeL            = beerVolumeL;
    this._beerStartTempC         = beerStartTempC;
    this._externalStartTempC     = externalStartTempC;
    this._targetTempC            = targetTempC;
    this._toleranceC             = toleranceC;
    this._ascM                   = ascM;
    this._runtimeM               = runtimeM;
    this._stepTimeS              = stepTimeS;
    this._chillerVolumeM3        = chillerVolumeM3;
    this._chillerOutputW         = chillerOutputW;
    this._heaterOutputW          = heaterOutputW;
    this._heaterRiseTimeS        = heaterRiseTimeS;
    this._heaterContactP         = heaterContactP;
    this._fermentationHeatW      = fermentationHeatW;
    this._thermalConductivityWmC = thermalConductivityWmC;
    this._surfaceAreaM2          = surfaceAreaM2;
    this._surfaceThicknessM      = surfaceThicknessM;
    this._surfaceExposedP        = surfaceExposedP;
        
    // Calculate how many simulation steps to run.
    this._numSteps = (runtimeM * 60) / stepTimeS;
        
    this._heater = new Heater(heaterOutputW, heaterRiseTimeS, stepTimeS);
    this._chiller = new Chiller(chillerVolumeM3, chillerOutputW, externalStartTempC, stepTimeS);
    this._fermenter = new Fermenter(beerVolumeL, beerStartTempC, fermentationHeatW, thermalConductivityWmC, surfaceAreaM2, surfaceThicknessM, surfaceExposedP, stepTimeS);
    this._controller = new Controller(targetTempC, toleranceC, ascM, stepTimeS);
        
    this._currentBeerTemp = beerStartTempC;
    
    this._beerTempData = [];
    this._heaterOutputData = [];
    this._chillerOutputData = [];
};


Simulator.prototype.getBeerTempData = function() {
    return this._beerTempData;
}

Simulator.prototype.getHeaterOutputData = function() {
    return this._heaterOutputData;
}

Simulator.prototype.getChillerOutputData = function() {
    return this._chillerOutputData;
}

Simulator.prototype.getPidOutputData = function() {
    return this._controller.getPidOutputData();
}

Simulator.prototype.getTargetTemp = function() {
    return this._targetTempC;
}

Simulator.prototype.getTolerance = function() {
    return this._toleranceC;
}


Simulator.prototype.run = function(cb) {
    console.log('Simulator::run');
    
    this._beerTempData = [];
    var context = {
        step: 0,
        numSteps: this._numSteps,
        stepsPerInterval: this._numSteps / 100,
        timerProgress: null,
        timerTask: null
    };
    var me = this;
    
    context.timerProgress = setInterval(function() {
        var value = (context.step / context.numSteps * 100).toFixed(0);
        $('#progress').css('width', (value)+'%').html((value)+'%');
    }, 250);
    
    context.timerTask = setInterval(function() {
        for (var ii=0; ii<context.stepsPerInterval && context.step<context.numSteps; ++ii, ++context.step) {
            me._controller.onStep(me._currentBeerTemp);
            me._updateInternalState();
        }
        
        if (context.step >= context.numSteps) {
            clearInterval(context.timerProgress);
            clearInterval(context.timerTask);
            $('#progress').css('width', '100%').html('graphing...');
            
            setTimeout(function() {
                $('#progress').css('width', '100%').html('done');
                cb();
            }, 100);
        }
    }, 10);
}


Simulator.prototype._updateInternalState = function() {

    this._heater.updateState( this._controller.heatOn() );    
    var heaterOutputFermenterW = this._heater.getOutputPower() * this._heaterContactP; // Adjust for partial contact w/ fermenter
    var heaterOutputAirW = this._heater.getOutputPower() * (1.0 - this._heaterContactP);
    
    this._chiller.updateState( this._controller.coolOn() );
    this._chiller.addEnergy( heaterOutputAirW );
    var externalTempC = this._chiller.getCurrentTemp();
    
    this._fermenter.updateState( externalTempC );
    this._fermenter.addEnergy(heaterOutputFermenterW);
    
    this._currentBeerTemp = this._fermenter.getTemperature();
    
    this._beerTempData.push(this._currentBeerTemp);
    this._heaterOutputData.push(heaterOutputFermenterW);
    this._chillerOutputData.push(externalTempC);
}