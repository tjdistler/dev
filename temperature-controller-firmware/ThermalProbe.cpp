#include "ThermalProbe.h"

#define TEMP_READ_DELAY_MS  1 * 1000


ThermalProbe::ThermalProbe(int pin, bool parasitic) :
    _bus(pin),
    _parasitic(parasitic ? 1 : 0),
    _inProgress(false),
    _error(false),
    _readTimer(TEMP_READ_DELAY_MS)
{
    _readTimer.disable();
}


void ThermalProbe::loop()
{
    if (_readTimer.enabled() && _readTimer.expired())
    {
        _completeRead();
    }
}


bool ThermalProbe::readBegin()
{
    if (_inProgress)
        return false;

    _temperature.setRaw( Temperature::UNDEFINED );
    _error = false;

    // Search for a device on the bus.
    if ( !_bus.search(_probeAddr))
    {
        _error = true;
        _bus.reset_search();
        Serial.println(String::format("No probe found on bus %d", _bus));
        return false;
    }
    _bus.reset_search();
    
    // Check CRC.
    if (OneWire::crc8(_probeAddr, 7) != _probeAddr[7])
    {
        Serial.println(String::format("Address CRC failure on bus %d", _bus));
        _error = true;
        return false;
    }

    // Check for supported chip type.
//    log("\tType: ");
    switch (_probeAddr[0]) 
    {
    case 0x10:
//        logln("DS1820/DS18S20");
        _probeType = 1;
        break;
    case 0x28:
//        logln("DS18B20");
        _probeType = 0;
        break;
    case 0x22:
//        logln("DS1822");
        _probeType = 0;
        break;
    case 0x26:
//        logln("DS2438");
        _probeType = 2;
        break;
    default:
        _error = true;
        return false;
    }
    
    // Read temperature
    _bus.reset();
    _bus.select(_probeAddr);
    _bus.write(0x44, _parasitic);
    
    _inProgress = true;
    _readTimer.restart();
    
    return true;
}


void ThermalProbe::_completeRead()
{
    _readTimer.disable();
    
    if (!_inProgress)
        return;

    _bus.reset();
    _bus.select(_probeAddr);
    _bus.write(0xB8, _parasitic);        // Recall Memory 0
    _bus.write(0x00, _parasitic);        // Recall Memory 0
    
    delay(20);
    _bus.reset();
    _bus.select(_probeAddr);
    _bus.write(0xBE, _parasitic);        // Read Scratchpad
    if (_probeType == 2) {
        _bus.write(0x00, _parasitic);    // The DS2438 needs a page# to read
    }
    
    byte data[12];
    for (int ii=0; ii<9; ++ii)
    {
        data[ii] = _bus.read();
    }
    
    // Check CRC
    if (OneWire::crc8(data, 8) != data[8])
    {
        Serial.println(String::format("Data CRC failure on bus %d", _bus));
        _error = true;
        _inProgress = false;
        return;
    }

    // Convert the data to actual temperature
    // because the result is a 16 bit signed integer, it should
    // be stored to an "int16_t" type, which is always 16 bits
    // even when compiled on a 32 bit processor.
    int16_t raw = (data[1] << 8) | data[0];
    if (_probeType == 2)
        raw = (data[2] << 8) | data[1];
    byte cfg = (data[4] & 0x60);
    
    switch (_probeType) {
    case 1:
        raw = raw << 3; // 9 bit resolution default
        if (data[7] == 0x10)
        {
            // "count remain" gives full 12 bit resolution
            raw = (raw & 0xFFF0) + 12 - data[6];
        }
        _temperature.setRaw( CELSIUS_TO_CELSIUS_T( (float)raw * 0.0625 ) );
        break;
    case 0:
        // at lower res, the low bits are undefined, so let's zero them
        if (cfg == 0x00) raw = raw & ~7;  // 9 bit resolution, 93.75 ms
        if (cfg == 0x20) raw = raw & ~3; // 10 bit res, 187.5 ms
        if (cfg == 0x40) raw = raw & ~1; // 11 bit res, 375 ms
        // default is 12 bit resolution, 750 ms conversion time
        _temperature.setRaw( CELSIUS_TO_CELSIUS_T( (float)raw * 0.0625 ) );
        break;
    
    case 2:
        data[1] = (data[1] >> 3) & 0x1f;
        if (data[2] > 127)
        {
            _temperature.setRaw( CELSIUS_TO_CELSIUS_T( (float)data[2] - ((float)data[1] * .03125) ) );
        }
        else
        {
            _temperature.setRaw( CELSIUS_TO_CELSIUS_T( (float)data[2] + ((float)data[1] * .03125) ) );
        }
    }
    
    Serial.println(String::format("\tTemperature: %3.4fC (%3.4fF), pin: %d", 
        CELSIUS_T_TO_CELSIUS(_temperature.getRaw()), CELSIUS_T_TO_FAHRENHEIT(_temperature.getRaw()), _bus));
    _inProgress = false;
    _error = false;
    return;
}