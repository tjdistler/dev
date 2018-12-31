#include "Hardware.h"


/**************************************
 * LED Object
 **************************************/
LED::LED(const int pin, const int activeState) :
    _pin(pin),
    _activeState(activeState),
    _level(1),
    _isOn(false)
{
    pinMode(_pin, OUTPUT);
    off();
}

void LED::loop()
{
    if (_timer.enabled() && _timer.expired())
    {
        if (_isOn)
            _off();
        else
            _on();
    }
}


void LED::setLevel(const float level)
{
    _level = level;
    if (_level > 1.0)
        _level = 1.0;
    else if (_level < 0.0)
        _level = 0.0;
        
    if (_isOn)
        _on();
}

    
void LED::on(const float level)
{
    if (level != -1)
        setLevel(level);
    _on();
    _timer.disable();
}


void LED::off()
{
    _off();
    _timer.disable();
}

void LED::blink(const unsigned long cycleTimeMS)
{
    unsigned long interval = cycleTimeMS / 2;
    if (interval < 10)
        interval = 10;
    _timer.setInterval(interval);
    _timer.enable();
}


unsigned LED::_levelToPWM(const float level) const
{
    if (_onPinValue() == HIGH) 
        return MAX_LEVEL * (level * level); // x^2 for exponential brightness curve
    return MAX_LEVEL - (MAX_LEVEL * (level * level));
}

void LED::_on()
{
    analogWrite(_pin, _levelToPWM(_level), LED_PWM_HZ);
    _isOn = true;
}

void LED::_off()
{
    digitalWrite(_pin, _offPinValue());
    _isOn = false;
}




/**************************************
 * Hardware Object
 **************************************/
Hardware::Hardware() :
    _probes({ {DS_BUS_0_PIN,DS_BUS_0_PARASITIC}, {DS_BUS_1_PIN,DS_BUS_1_PARASITIC} }),
    _leds({ {RED_LED_PIN,    LED_ON_STATE}, 
            {BLUE_LED_PIN,   LED_ON_STATE},
            {STATUS_LED_PIN, LED_ON_STATE} }),
    _lcd(LCD_ADDR, LCD_COLUMNS, LCD_ROWS),
    _encoder(ENCODER_A_PIN, ENCODER_B_PIN, ENCODER_BTN_PIN)
{
}
    
void Hardware::setup()
{
    Serial.begin(SERIAL_BAUD);
    
    pinMode(RELAY_0_PIN, OUTPUT);
    pinMode(RELAY_1_PIN, OUTPUT);

    relayOpen(HEAT_RELAY);
    relayOpen(COOL_RELAY);
    
    _encoder.setup();
    
    // Allow time for PWM to settle.
    delay(100);
}


void Hardware::loop()
{
    _encoder.loop();
    
    for (int ii=0; ii<3; ++ii)
        _leds[ii].loop();
        
    for (int ii=0; ii<2; ++ii)
        _probes[ii].loop();
}


void Hardware::relayClose(const relay_t relay)
{
    if (relay == HEAT_RELAY)
        digitalWrite(RELAY_0_PIN, RELAY_CLOSE_STATE);
    else
        digitalWrite(RELAY_1_PIN, RELAY_CLOSE_STATE);
}


void Hardware::relayOpen(const relay_t relay)
{
    if (relay == HEAT_RELAY)
        digitalWrite(RELAY_0_PIN, RELAY_CLOSE_STATE == HIGH ? LOW : HIGH);
    else
        digitalWrite(RELAY_1_PIN, RELAY_CLOSE_STATE == HIGH ? LOW : HIGH);
}
