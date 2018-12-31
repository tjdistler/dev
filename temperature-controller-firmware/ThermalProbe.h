#ifndef _THERMAL_PROBE_H_
#define _THERMAL_PROBE_H_

#include <OneWire.h>
#include "Common.h"

class ThermalProbe
{
public:
    ThermalProbe(int pin, bool parasitic);
    
    void loop();
    
    bool readBegin();
    bool complete(bool &error) const { error = _error; return !_inProgress; }
    Temperature getTemperature() const { return _temperature; }
    
private:
    void _completeRead();
    
    OneWire _bus;
    int _parasitic;
    bool _inProgress;
    bool _error;
    byte _probeAddr[8];
    byte _probeType;
    Temperature _temperature;
    PolledTimer _readTimer;
};

#endif