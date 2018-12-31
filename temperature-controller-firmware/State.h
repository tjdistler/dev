#ifndef _STATE_H_
#define _STATE_H_

#include "Common.h"
#include "Config.h"


class State
{
public:
    typedef enum {
        ALL_OFF = 0,
        HEAT_ASC = 1,
        HEAT_ON = 2,
        COOL_ASC = -1,
        COOL_ON = -2
    } output_state_t;
    
    State(const output_state_t state);
    
    unsigned long version() const { return _version; }
    
    void setFormat(const Temperature::format_t format) { _format = format; }
    Temperature::format_t getFormat() const { return _format; }
    
    void setProbeTemp(const int index, const Temperature value);
    bool probeTempReady(const int index) const { return _smoothedTemps[index].ready(); }
    Temperature getProbeTemp(const int index) const { return _smoothedTemps[index].get(); }
    Temperature getProbeTempRaw(const int index) const { return _rawTemps[index]; }
    
    void setOutputState(const output_state_t state) { ++_version; _outputState = state; }
    output_state_t getOutputState() const { return _outputState; }
    
    void setAverageTemp(const Temperature value) { ++_version; _avgTemp = value; }
    Temperature getAverageTemp() const { return _avgTemp; }
    
    void setPIDOutput(const float output) { ++_version; _pidOutput = output; }
    float getPIDOutput() const { return _pidOutput; }
    
private:
    unsigned long _version;
    Temperature::format_t _format;
    Temperature _rawTemps[2];
    SimpleMovingAverage<Temperature,PROBE_WINDOW_SIZE> _smoothedTemps[2];
    output_state_t _outputState;
    Temperature _avgTemp;
    float _pidOutput;
};

#endif