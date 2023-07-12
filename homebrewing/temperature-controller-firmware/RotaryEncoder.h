#ifndef _ROTARY_ENCODER_H_
#define _ROTARY_ENCODER_H_

#include "Common.h"

class RotaryEncoder
{
public:
    RotaryEncoder(int pinA, int pinB, int pinButton);
    
    void setup();
    void loop();
    
    bool hasNewData() const { return _newData; }
    
    int getCount() const { return _count; }
    
    bool buttonPressed() const { return _buttonPressed; }
    
    void resetData() { 
        _newData = false;
        _count = 0;
        _buttonPressed = false;
        _fastRotationCount = 0;
    }
    
private:
    const int _pinA, _pinB, _pinButton;
    bool _newData;  // Indicates if new input has been received.
    int _count;     // Accumulated rotational count
    bool _buttonPressed;
    Timer _readTimer;
    PolledTimer _runTimer; // Only poll pins for a certain length of time after interrupt.
    int _pinAPrevState, _pinBPrevState;
    unsigned long _lastRotationMs;
    int _fastRotationCount;
    Debounce<int,ENCODER_BUTTON_DEBOUNCE_COUNT> _buttonDebounce;
    int _prevButtonState;
    
    
    void _isr();
    void _read();
};

#endif