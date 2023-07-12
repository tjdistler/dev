#include "State.h"


State::State(const output_state_t state) :
    _version(0),
    _outputState(state),
    _pidOutput(0)
{
}


void State::setProbeTemp(const int index, const Temperature value)
{
    ++_version;
    
    if (!value.isValid()) {
        _rawTemps[index] = value;
        _smoothedTemps[index].reset(value);
        return;
    }
    
    _rawTemps[index] = value;
    _smoothedTemps[index].add(value);
}
