#ifndef _HARDWARE_H_
#define _HARDWARE_H_

#include <LiquidCrystal_I2C_Spark.h>
#include "Common.h"
#include "ThermalProbe.h"
#include "RotaryEncoder.h"


/**************************************
 * LED Object
 **************************************/
class LED
{
public:
    static const unsigned MAX_LEVEL = 255;
    
    LED(const int pin, const int activeState);
    
    void loop();
    
    // 0-1.0
    void setLevel(const float level);
    void on(const float level = -1);
    void off();
    bool isOn() const { return _isOn; }
    void blink(const unsigned long cycleTimeMS);
    
private:
    int _pin;
    int _activeState;   // i.e. pin state to turn LED on
    float _level;    // 0-1, max level of pulse
    PolledTimer _timer;
    bool _isOn;
    
    int _onPinValue() const { return _activeState; }
    int _offPinValue() const { return _activeState == HIGH ? LOW : HIGH; }
    
    // 0.0-1.0 -> 0-255
    unsigned _levelToPWM(const float level) const;
    void _on();
    void _off();
};



/**************************************
 * Hardware Object
 **************************************/
class Hardware
{
public:

    typedef enum {
        RED_LED = 0,
        BLUE_LED,
        STATUS_LED
    } led_t;
    
    typedef enum {
        HEAT_RELAY,
        COOL_RELAY
    } relay_t;
    
    Hardware();
    
    void setup();
    void loop();
    
    ThermalProbe& probe(int bus) { return _probes[bus]; }
    
    LED& led(const led_t led) { return _leds[led]; }
    
    LiquidCrystal_I2C& lcd() { return _lcd; }
    
    RotaryEncoder& encoder() { return _encoder; }
    
    void relayClose(const relay_t relay);
    void relayOpen(const relay_t relay);
    
private:
    ThermalProbe _probes[2];
    LED _leds[3];
    LiquidCrystal_I2C _lcd;
    RotaryEncoder _encoder;
};

#endif