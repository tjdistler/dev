#ifndef _CONTROLLER_H_
#define _CONTROLLER_H_

#include "Common.h"
#include "Config.h"
#include "State.h"


// Makes output decisions based on the exponential moving average of the input.
class Controller
{
public:
    typedef enum {
        OFF,
        HEAT_ASC,
        HEAT,
        COOL_ASC,
        COOL
    } decision_t;
    
    Controller();
    
    void setup(Config* config, State* state);
    
    decision_t compute(const Temperature currentTemp);
    
private:
    Config* _config;
    State* _state;
    unsigned long _configVersion;
    bool _waitForStability;
    celsius_t _pidIntegralTerm;
    celsius_t _prevProcessVar;
    decision_t _currentDecision;
    PolledTimer _ascTimer;
    SimpleMovingAverage<Temperature, TEMP_AVERAGE_WINDOW_SAMPLES> _averageTemp;
};

#endif