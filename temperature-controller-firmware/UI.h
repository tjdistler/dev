#ifndef _UI_H_
#define _UI_H_

#include "Common.h"
#include "Hardware.h"
#include "State.h"
#include "Config.h"

class UI
{
    // UI states for state-machine
    typedef enum {
        MAIN,
        SPLASH,
        UPDATING
    } ui_state_t;
    
    // States for controlling the status LED
    typedef enum {
        NO_WIFI,
        NO_INTERNET,
        CONNECTED
    } status_led_state_t;
    
public:
    UI(const unsigned long updateIntervalMs);
    
    void setup(Hardware* hw, Config* config, State* state);
    void loop();
    
    void onFirmwareUpdate(system_event_t event, int param, void* data);
    
private:
    ui_state_t _uiState;
    status_led_state_t _statusLEDState;
    Hardware* _hw;
    Config* _config;
    State* _state;
    PolledTimer _updateTimer;
    LiquidCrystal_I2C* _lcd;
    PolledTimer _backlightTimer;
    bool _lcdOn;
    RotaryEncoder* _encoder;
    
    void _onStateMain();
    void _onStateSplash();
    
    void _updateStatusLEDsAndIcons();
    void _processInputs();
    
    void _displayOn();
    void _displayOff();
};

#endif