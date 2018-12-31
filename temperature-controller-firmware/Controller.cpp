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
    
    _state->setAutoSetpoint( _config->getSetpoint() );
    
    _configVersion = _config->version();
    
    _averageTemp.reset(Temperature::UNDEFINED);
    
    _ascTimer.setInterval(_config->getAsc());
    _ascTimer.restart();
}
    

Controller::decision_t Controller::compute(const Temperature currentTemp)
{
    // Reset parameters if config has changed
    if (_configVersion != _config->version())
    {
        _averageTemp.reset(Temperature::UNDEFINED);
        _prevProcessVar = Temperature::UNDEFINED;
        _pidIntegralTerm = 0;
        _pidDerivativeSet.reset();
        _ascTimer.setInterval(_config->getAsc());
        _waitForStability = true;
        _configVersion = _config->version();
    }
    
    celsius_t setpoint = _config->getSetpoint().getRaw();
    if (_config->getAutoAdjustEnabled())
    {
        // Dynamically calculate the setpoint if auto-adjust is enabled
        float deltaTime = Time.now() - _config->getAutoAdjustStartTS();
        
        float percent = deltaTime / (float)_config->getAutoTimePeriod();
        if (percent >= 1.0)
        {
            // Auto-adjust is complete, so set the new setpoint and disable auto-adjust
            _config->setSetpoint( _config->getAutoSetpoint() );
            _config->setAutoAdjustEnabled( false );
            _config->setAutoAdjustStartTS( 0 );
            percent = 1.0; // limit changes to 100%
        }
        
        Temperature deltaTemp = _config->getAutoSetpoint().getRaw() - _config->getSetpoint().getRaw();
        setpoint = setpoint + (deltaTemp.getRaw() * percent);
        
        logToCloud(String::format("dtmp:%d, dt:%f, p:%f, st:%d", deltaTemp.getRaw(), deltaTime, percent, setpoint));
    }
    
    // Update the dynamic state (not to be confused with the config)
    _state->setAutoSetpoint(setpoint);

    const celsius_t current = currentTemp.getRaw();
    const celsius_t tolerance = _config->getTolerance().getRaw();
    const celsius_t toleranceMax = setpoint + tolerance;
    const celsius_t toleranceMin  = setpoint - tolerance;
    
    // Only perform PID adjustments if the temperture is inside the tolerance range.
    if (current <= toleranceMax && current >= toleranceMin)
        _waitForStability = false;

    float controlVariable = 0;
    celsius_t processVariable = Temperature::UNDEFINED;
    float P = 0;
    float I = 0;
    float D = 0;
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
            P = processVariable - setpoint;
            
            if (_prevProcessVar == Temperature::UNDEFINED)
                _prevProcessVar = processVariable;
        
            // Integral
            if (_config->getPidKi() != 0)
            {
                _pidIntegralTerm += P;
                I = _pidIntegralTerm / _config->getPidKi();
            }
                
            // Derivative
            _pidDerivativeSet.add(P);
            D = _config->getPidKd() * (_pidDerivativeSet.newest() - _pidDerivativeSet.oldest());
        
            controlVariable = _config->getPidKp() * ( P + I + D );
            _prevProcessVar = processVariable;
            
            // Limit output to a percentage of the tolerance range
            controlVariable = range<float>(controlVariable, tolerance * TOLERANCE_PERCENTAGE_LIMIT, tolerance * -TOLERANCE_PERCENTAGE_LIMIT);
            
            _state->setAverageTemp(_averageTemp.get());
            _state->setPIDOutput(controlVariable);
        }
    }
    
    const celsius_t adjustedSetpoint = setpoint - controlVariable;
    const celsius_t toleranceHigh = range(adjustedSetpoint + tolerance, toleranceMax, toleranceMin);
    const celsius_t toleranceLow  = range(adjustedSetpoint - tolerance, toleranceMax, toleranceMin);

    Serial.printf("PID: raw: %d, pv: %d, setpoint: %d, tolerance: %d, cv: %.2f, adjusted: %d, P: %.2f, I: %.2f, D: %.2f\r\n",
        current, processVariable, setpoint, tolerance, controlVariable, adjustedSetpoint, P, I, D);
    
    logToCloud(String::format("PID: raw: %d, pv: %d, setpoint: %d, tolerance: %d, cv: %.2f, adjusted: %d, P: %.2f, I: %.2f, D: %.2f",
        current, processVariable, setpoint, tolerance, controlVariable, adjustedSetpoint, P, I, D));
    
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