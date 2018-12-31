#include "Controller.h"


Controller::Controller() :
    _config(0),
    _state(0),
    _configVersion(-1),
    _waitForStability(true),
    _pidIntegralTerm(0),
    _prevProcessVar(Temperature::UNDEFINED),
    _currentDecision(OFF)
{
}
    

void Controller::setup(Config* config, State* state)
{
    _config = config;
    _state  = state;
    
    _configVersion = _config->version();
    
    _ascTimer.setInterval(_config->getAsc());
    _ascTimer.restart();
}
    

Controller::decision_t Controller::compute(const Temperature currentTemp)
{
    // Reset parameters if config has changed
    if (_configVersion != _config->version())
    {
        _averageTemp.reset();
        _prevProcessVar = Temperature::UNDEFINED;
        _pidIntegralTerm = 0;
        _ascTimer.setInterval(_config->getAsc());
        _waitForStability = true;
        _configVersion = _config->version();
    }

    const celsius_t current = currentTemp.getRaw();
    const celsius_t setpoint = _config->getSetpoint().getRaw();
    const celsius_t tolerance = _config->getTolerance().getRaw();
    const celsius_t toleranceMax = setpoint + tolerance;
    const celsius_t toleranceMin  = setpoint - tolerance;
    
    // Only perform PID adjustments if the temperture is inside the tolerance range.
    if (current <= toleranceMax && current >= toleranceMin)
        _waitForStability = false;

    float controlVariable = 0;
    celsius_t processVariable = Temperature::UNDEFINED;
    if (!_waitForStability)
    {
        // Calculate moving average and use as the PID process variable.
        _averageTemp.add(current);
        if (_averageTemp.ready())
        {
            // PID
            // Kp = gain
            // Ki = reset
            // Ideal PID algorithm: Kp * [ e(t) + 1/Ki * integral(e(t)d(t)) + Kd * d*e(t)/d(t)]
            
            processVariable = _averageTemp.get().getRaw();
            const float error = processVariable - setpoint;
            
            if (_prevProcessVar == Temperature::UNDEFINED)
                _prevProcessVar = processVariable;
        
            _pidIntegralTerm += error;
            
            //TODO: Limit integral term???
        
            float I = 0;
            if (_config->getPidKi() != 0)
                I = (1.0 / _config->getPidKi()) * _pidIntegralTerm;
        
            controlVariable = _config->getPidKp() * ( error + I ); // + (_config->getPidKd() * (current - _prevProcessVar));
            _prevProcessVar = processVariable;
            
            // Limit output to tolerance range
            controlVariable = range<float>(controlVariable, tolerance, -1 * tolerance);
            
            _state->setAverageTemp(_averageTemp.get());
            _state->setPIDOutput(controlVariable);
        }
    }
    
    const celsius_t adjustedSetpoint = setpoint - controlVariable;
    const celsius_t toleranceHigh = range(adjustedSetpoint + tolerance, toleranceMax, toleranceMin);
    const celsius_t toleranceLow  = range(adjustedSetpoint - tolerance, toleranceMax, toleranceMin);

    Serial.printf("PID: pv: %d, setpoint: %d, tolerance: %d, cv: %.2f, adjusted: %d\r\n",
        processVariable, setpoint, tolerance, controlVariable, adjustedSetpoint);
    
    // Update decision
    switch (_currentDecision)
    {
    case OFF:
    {
        if (current <= toleranceLow)
            _currentDecision = HEAT_ASC;
        else if (current >= toleranceHigh)
            _currentDecision = COOL_ASC;
        break;
    }
    case HEAT_ASC:
    {
        if (!_ascTimer.expired(false))
        {
            if (current > toleranceLow)
                _currentDecision = OFF;
            else if (current >= toleranceHigh)
                _currentDecision = COOL_ASC;
            break;
        }
        else
        {
            _currentDecision = HEAT;
            // let fall-through to HEAT state.
        }
    }
    case HEAT:
    {
        if (current >= adjustedSetpoint)
        {
            _currentDecision = OFF;
            _ascTimer.restart();
        }
        break;
    }
    case COOL_ASC:
    {
        if (!_ascTimer.expired(false))
        {
            if (current < toleranceHigh)
                _currentDecision = OFF;
            else if (current <= toleranceLow)
                _currentDecision = HEAT_ASC;
            break;
        }
        else
        {
            _currentDecision = COOL;
            // let fall-through to COOL state.
        }
    }
    case COOL:
    {
        if (current <= adjustedSetpoint)
        {
            _currentDecision = OFF;
            _ascTimer.restart();
        }
        break;
    }
    };
    
    // Respect configured overrides
    if ((_currentDecision == HEAT_ASC || _currentDecision == HEAT) && !_config->getHeatEnabled())
        _currentDecision = OFF;
    if ((_currentDecision == COOL_ASC || _currentDecision == COOL) && !_config->getCoolEnabled())
        _currentDecision = OFF;
    
    return _currentDecision;
}