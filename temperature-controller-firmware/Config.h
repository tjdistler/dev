#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "Common.h"

// A class for managing the configuration data, including load/store.
class Config
{
    typedef struct
    {
        Temperature setpoint;
        Temperature tolerance;
        unsigned long asc;      // ms
        Temperature offsets[2];
        unsigned char ledLevel; // 0-100
        bool heatEnabled;
        bool coolEnabled;
        float Kp;
        float Ki;
        float Kd;
    } config_t;

public:
    Config();
    
    Config( const Temperature setpoint,
            const Temperature tolerance,
            const unsigned long ascMs,
            const Temperature offset0,
            const Temperature offset1,
            const unsigned char ledLevel,
            const bool heatEnabled,
            const bool coolEnabled,
            const float Kp,
            const float Ki,
            const float Kd);
    
    Config& operator=(const Config& other)
    {
        memcpy(&_config, &other._config, sizeof(_config));
        _version = other._version;
        return *this;
    }
    
    void load();    // Load values from flash
    void store();   // Save values to flash
    
    unsigned long version() const { return _version; }
    
    Temperature getSetpoint() const { return _config.setpoint; }
    void setSetpoint(Temperature value) { ++_version; _config.setpoint = value; }
    
    Temperature getTolerance() const { return _config.tolerance; }
    void setTolerance(Temperature value) { ++_version; _config.tolerance = value; }
    
    unsigned long getAsc() const { return _config.asc; }
    void setAsc(unsigned long value) { ++_version; _config.asc = value; }
    
    Temperature getProbeOffset(size_t index) const { return _config.offsets[index]; }
    void setProbeOffset(size_t index, Temperature value) { ++_version; _config.offsets[index] = value; }
    
    unsigned char getLEDLevel() const { return _config.ledLevel; }
    void setLEDLevel(unsigned char level) { ++_version; _config.ledLevel = level; }
    
    bool getHeatEnabled() const { return _config.heatEnabled; }
    void setHeatEnabled(bool enabled) { ++_version; _config.heatEnabled = enabled; }
    
    bool getCoolEnabled() const { return _config.coolEnabled; }
    void setCoolEnabled(bool enabled) { ++_version; _config.coolEnabled = enabled; }
    
    float getPidKp() const { return _config.Kp; }
    void setPidKp(float value) { ++_version; _config.Kp = value; }
    
    float getPidKi() const { return _config.Ki; }
    void setPidKi(float value) { ++_version; _config.Ki = value; }
    
    float getPidKd() const { return _config.Kd; }
    void setPidKd(float value) { ++_version; _config.Kd = value; }
    
private:
    config_t _config;
    unsigned long _version; // incremented on every update.
};


#endif