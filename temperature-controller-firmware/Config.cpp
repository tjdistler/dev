#include "Config.h"


const size_t BASE_ADDR = 0;
const uint32_t FLASH_MAGIC = 0xcc3cf2e9; // Magic number indicating if Flash data is valid.


/**************************************
 * Flash
 **************************************/
template<class T>
  void writeToFlash(int address, const T& value)
{
  const size_t numBytes = sizeof(T);
  const uint8_t* p = (uint8_t*)&value;
  for (size_t ii=0; ii<numBytes; ++ii)
    EEPROM.write(address+ii, p[ii]);
}


template<class T>
  void readFromFlash(int address, T& value)
{
  const size_t numBytes = sizeof(T);
  uint8_t* p = (uint8_t*)&value;
  for (size_t ii=0; ii<numBytes; ++ii)
    p[ii] = EEPROM.read(address+ii);
}


/**************************************
 * Config Object
 **************************************/
Config::Config() :
    _config({DEFAULT_SETPOINT,
             DEFAULT_TOLERANCE,
             DEFAULT_ASC,
             { 0, 0 },
             100,
             true,
             true,
             DEFAULT_KP,
             DEFAULT_KI,
             DEFAULT_KD }),
    _version(0)
{
}

Config::Config( const Temperature setpoint,
                const Temperature tolerance,
                const unsigned long ascMs,
                const Temperature offset0,
                const Temperature offset1,
                const unsigned char ledLevel,
                const bool heatEnabled,
                const bool coolEnabled,
                const float Kp,
                const float Ki,
                const float Kd) :
    _config({setpoint, tolerance, ascMs, {offset0, offset1}, ledLevel, heatEnabled, coolEnabled, Kp, Ki, Kd}),
    _version(0)
{
}

    
void Config::load()
{
    uint32_t magicRead = 0;
    size_t addr = BASE_ADDR;
    readFromFlash(addr, magicRead);
    if (magicRead != FLASH_MAGIC)
    {
        store();
        return;
    }

    readFromFlash(addr + sizeof(FLASH_MAGIC), _config);
}


void Config::store()
{
    size_t addr = BASE_ADDR;
    writeToFlash(addr, FLASH_MAGIC);
    writeToFlash(addr + sizeof(FLASH_MAGIC), _config);
}