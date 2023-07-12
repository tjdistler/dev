#include "UI.h"


UI::UI(const unsigned long updateIntervalMs) :
    _uiState(MAIN),
    _statusLEDState(NO_WIFI),
    _hw(0),
    _config(0),
    _state(0),
    _updateTimer(updateIntervalMs),
    _lcd(0),
    _backlightTimer(UI_BACKLIGHT_ON_MS),
    _lcdOn(false),
    _encoder(0)
{
}
    
void UI::setup(Hardware* hw, Config* config, State* state)
{
    _hw = hw;
    _config = config;
    _state = state;
    _lcd = &(hw->lcd());
    _encoder = &(hw->encoder());
    
    LED& statusLED = _hw->led(Hardware::STATUS_LED);
    statusLED.blink(UI_NO_WIFI_LED_BLINK_MS);
    
    _displayOn();
    
    _updateTimer.restart();
    _backlightTimer.restart();
}


void UI::loop()
{
    // Turn off display after a time.
    if (_backlightTimer.enabled() && _backlightTimer.expired())
    {
        _displayOff();
        return;
    }
    
    _processInputs();
    
    
    if (_updateTimer.expired())
    {
        _updateStatusLEDsAndIcons();
        
        switch (_uiState)
        {
        case MAIN:
            _onStateMain();
            break;
        case SPLASH:
            _onStateSplash();
            break;
        case UPDATING:
            break;
        };
    }
}


void UI::_onStateMain()
{
    // TODO: inputs
    
    if (!_lcdOn)
        return;
        
    int idx = 0;
    String text;
    Temperature temp0 = _state->getProbeTempRaw(0);
    Temperature temp1 = _state->getProbeTempRaw(1);
        
    // Row 0
    text = "";
    _lcd->setCursor(0, 0);
    if (temp0.isValid())
        text += String::format("%03.1f%c", temp0.get(_state->getFormat()), DEGREE_CHAR);
    else
        text += String::format("--.-%c", DEGREE_CHAR);
        
    if (temp1.isValid())
        text += String::format(" [%03.1f%c]", temp1.get(_state->getFormat()), DEGREE_CHAR);
    else
        text += String::format(" [--.-%c]", DEGREE_CHAR);
        
    // Clear the rest of the top row (leave room for status icons).
    idx += text.length();
    for (int ii=idx; ii<LCD_COLUMNS-4; ++ii, ++idx)
        text += " ";
    _lcd->print(text);
      
        
    // Row 1
    _lcd->setCursor(0, 1);
    
    
    // Row 2
    text = "";
    _lcd->setCursor(0, 2);
    text = String::format("SP: %03.1f%c ASC:%dm", 
        _config->getSetpoint().get(_state->getFormat()), 
        DEGREE_CHAR,
        _config->getAsc() / 1000 / 60);

    idx += text.length();
    for (int ii=idx; ii<LCD_COLUMNS; ++ii, ++idx)
        text += " ";
    _lcd->print(text);
    
    
    // Row 3
    text = "";
    _lcd->setCursor(0, 3);
    if (_state->getAverageTemp().isValid())
        text = String::format("Avg:%03.1f%c", _state->getAverageTemp().get(_state->getFormat()), DEGREE_CHAR);
    else
        text = String::format("Avg:--.-%c", DEGREE_CHAR);
        
    text += " ST: ";
    switch (_state->getOutputState())
    {
    case State::ALL_OFF:
        text += "Off  "; break;
    case State::HEAT_ASC:
    case State::COOL_ASC:
        text += "ASC  "; break;
    case State::HEAT_ON:
        text += "Heat "; break;
    case State::COOL_ON:
        text += "Cool "; break;
    default:
        text += "Err";
    };

    idx += text.length();
    for (int ii=idx; ii<LCD_COLUMNS; ++ii, ++idx)
        text += " ";
    _lcd->print(text);
}


void UI::_onStateSplash()
{
    
}


void UI::_updateStatusLEDsAndIcons()
{
    if (_uiState == UPDATING)
        return;

    // Status LED
    LED& statusLED = _hw->led(Hardware::STATUS_LED);
    if (!WiFi.ready())
    {
        if (_statusLEDState != NO_WIFI)
            statusLED.blink(UI_NO_WIFI_LED_BLINK_MS);
        _statusLEDState = NO_WIFI;
    }
    else if (!Particle.connected())
    {
        if (_statusLEDState != NO_INTERNET)
            statusLED.blink(UI_NO_INTERNET_LED_BLINK_MS);
        _statusLEDState = NO_INTERNET;
    }
    else if (_statusLEDState != CONNECTED)
    {
        statusLED.on();
        _statusLEDState = CONNECTED;
    }
    
    // Heat/Cool LEDs
    LED& heatLED = _hw->led(Hardware::RED_LED);
    LED& coolLED = _hw->led(Hardware::BLUE_LED);
    switch (_state->getOutputState())
    {
    case State::ALL_OFF:
        heatLED.off();
        coolLED.off();
        break;
    case State::HEAT_ASC:
        coolLED.off();
        heatLED.blink(UI_ASC_BLINK_MS);
        break;
    case State::COOL_ASC:
        heatLED.off();
        coolLED.blink(UI_ASC_BLINK_MS);
        break;
    case State::HEAT_ON:
        coolLED.off();
        heatLED.on();
        break;
    case State::COOL_ON:
        heatLED.off();
        coolLED.on();
        break;
    default:
        heatLED.off();
        coolLED.off();
    }
    
    // Update status icons
    _lcd->setCursor(LCD_COLUMNS-4, 0);
    
    // Update probe presence
    _state->getProbeTempRaw(0).isValid() ? _lcd->print("+") : _lcd->print("_");
    _state->getProbeTempRaw(1).isValid() ? _lcd->print("+") : _lcd->print("_");
    
    // Update WiFi signal strength meter
    if (!WiFi.ready())
        _lcd->print("_");
    else
    {
        // RSSI: -1 to -127db (lower is worse)
        int rssi = WiFi.RSSI();
        if (rssi >= 0)
            _lcd->print("_");
        else
        {
            // We use bits 6 and 7 to determine what icon to display.
            int idx = (-rssi) >> 5;
            _lcd->write(WIFI_CHAR_ADDR[idx]);
        }
    }
    
    // Update cloud connectivity icon
    if (!Particle.connected())
        _lcd->print("_");
    else
        _lcd->write(CHECK_CHAR_ADDR);
}


void UI::_processInputs()
{
    // Wake up LCD if it's off.
    if (!_lcdOn)
    {
        if (_encoder->hasNewData())
        {
            _displayOn();
            _encoder->resetData(); // Throw input away
        }
        return;
    }
    
    
    if (_encoder->hasNewData())
    {
        _backlightTimer.restart();
        
        
        //TODO: Implement
        _encoder->resetData();
    }
}


void UI::onFirmwareUpdate(system_event_t event, int param, void* data)
{
    switch(param)
    {
    case firmware_update_begin:
    {
        _displayOn();
        _lcd->clear();
        _uiState = UPDATING;
        
        const String text = "Updating...";
        int col = (LCD_COLUMNS / 2) - (text.length() / 2);
        int row = (LCD_ROWS / 2) - 1;
        _lcd->setCursor(col, row);
        _lcd->print(text);
        
        _hw->led(Hardware::RED_LED).off();
        _hw->led(Hardware::BLUE_LED).off();
        _hw->led(Hardware::STATUS_LED).blink(UI_FIRWARE_UPDATE_BLINK_MS);
        break;
    }
    case firmware_update_progress:
        break;
    case firmware_update_complete:
    case firmware_update_failed:
        _displayOff();
        break;
    };
}


void UI::_displayOn()
{
    if (!_lcdOn)
    {
        _lcd->init();
        _lcd->createChar( WIFI_CHAR_ADDR[0], const_cast<uint8_t*>(WIFI_CHAR[0]) );
        _lcd->createChar( WIFI_CHAR_ADDR[1], const_cast<uint8_t*>(WIFI_CHAR[1]) );
        _lcd->createChar( WIFI_CHAR_ADDR[2], const_cast<uint8_t*>(WIFI_CHAR[2]) );
        _lcd->createChar( WIFI_CHAR_ADDR[3], const_cast<uint8_t*>(WIFI_CHAR[3]) );
        _lcd->createChar( CHECK_CHAR_ADDR,   const_cast<uint8_t*>(CHECK_CHAR) );
        _lcd->display();
        _lcd->backlight();
        _lcd->clear();
        _lcdOn = true;
        _backlightTimer.restart();
    }
}


void UI::_displayOff()
{
    _lcd->noDisplay();
    _lcd->noBacklight();
    _lcdOn = false;
    _backlightTimer.disable();
}