#include "RotaryEncoder.h"


// 0 = undefined, -1 = CCW, 1 = CW
// Index into table is the 4-bit value defined as:
//  Old   |  New
// A   B  | A   B
//
const int ENCODER_TABLE[16] = {
    0,  // 0 0 | 0 0
   -1,  // 0 0 | 0 1
    1,  // 0 0 | 1 0
    0,  // 0 0 | 1 1
    0,  // 0 1 | 0 0
    0,  // 0 1 | 0 1
    0,  // 0 1 | 1 0
    0,  // 0 1 | 1 1
    0,  // 1 0 | 0 0
    0,  // 1 0 | 0 1
    0,  // 1 0 | 1 0
    0,  // 1 0 | 1 1
    0,  // 1 1 | 0 0
    0,  // 1 1 | 0 1
    0,  // 1 1 | 1 0
    0,  // 1 1 | 1 1
};


RotaryEncoder::RotaryEncoder(int pinA, int pinB, int pinButton) :
    _pinA(pinA),
    _pinB(pinB),
    _pinButton(pinButton),
    _newData(false),
    _count(0),
    _buttonPressed(false),
    _readTimer(ENCODER_READ_INTERVAL_MS, &RotaryEncoder::_read, *this),
    _runTimer(ENCODER_ACTIVE_READ_DURATION_MS),
    _pinAPrevState(HIGH),
    _pinBPrevState(HIGH),
    _lastRotationMs(millis()),
    _fastRotationCount(0),
    _prevButtonState(ENCODER_BUTTON_RELEASED_STATE)
{
}


void RotaryEncoder::setup()
{
    pinMode(_pinA, INPUT_PULLUP);
    pinMode(_pinB, INPUT_PULLUP);
    pinMode(_pinButton, INPUT_PULLUP);
    
    attachInterrupt(_pinA, &RotaryEncoder::_isr, this, CHANGE);
    attachInterrupt(_pinB, &RotaryEncoder::_isr, this, CHANGE);
    attachInterrupt(_pinButton, &RotaryEncoder::_isr, this, CHANGE);
    
    _runTimer.disable();
}


void RotaryEncoder::loop()
{
    if (_runTimer.enabled() && _runTimer.expired())
    {
        _runTimer.disable();
        _readTimer.stop();
    }
}


void RotaryEncoder::_isr()
{
    // Got some user data, so start polling the encoder for awhile.
    if (_runTimer.disabled())
    {
        _readTimer.startFromISR();
        _runTimer.restart();
    }
}


void RotaryEncoder::_read()
{
    const int pinA = pinReadFast(_pinA);
    const int pinB = pinReadFast(_pinB);
    _buttonDebounce.add( pinReadFast(_pinButton) );

    bool input = false;
        
    if (pinA != _pinAPrevState || pinB != _pinBPrevState)
    {
    
        // Build index into truth table
        int idx = 0;
        idx |= _pinAPrevState == LOW ? 0x08 : 0;
        idx |= _pinBPrevState == LOW ? 0x04 : 0;
        idx |= pinA == LOW ? 0x02 : 0;
        idx |= pinB == LOW ? 0x01 : 0;
        
        // Handle "fast" scaling
        const unsigned long now = millis();
        if (now - _lastRotationMs < ENCODER_FAST_ROTATION_MIN_MS)
            ++_fastRotationCount;
        else
            _fastRotationCount = 0;
            
        if (_fastRotationCount < ENCODER_FAST_ROTATION_TICKS)
            _count += ENCODER_TABLE[idx];
        else
            _count += ENCODER_TABLE[idx] * ENCODER_FAST_SCALAR;
        
        _pinAPrevState = pinA;
        _pinBPrevState = pinB;
        _lastRotationMs = now;

        input = true;
    }
    
    if (_buttonDebounce.ready())
    {
        // Trigger "pressed" on low->high transition (i.e. button release)
        const int currentState = _buttonDebounce.get();
        if (_prevButtonState == ENCODER_BUTTON_PRESSED_STATE && currentState == ENCODER_BUTTON_RELEASED_STATE)
        {
            _buttonPressed = true;
            input = true;
        }

        _prevButtonState = currentState;
    }
    
    if (input)
    {
        _newData = true;
        _runTimer.restart();
    }
}