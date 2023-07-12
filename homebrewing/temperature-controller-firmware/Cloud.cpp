#include "Cloud.h"
#include <ThingSpeak.h>


int parseOptionIndexes(const String config, const String prefix, const char* delimiter, int* begin, int* end)
{
    // Make sure 'a:' doesn't resolve for 'ba:'. Search for ',a:' and '{a:'.
    String search = "," + prefix;
    *begin = config.indexOf(search);
    if (*begin == -1)
    {
        search = "{" + prefix;
        *begin = config.indexOf(search);
        if (*begin == -1)
            return 0;
    }

    *begin += search.length();
    *end = config.indexOf(delimiter, *begin);
    return 1;
}


// Parse out a config value.
//
// config - The full input config string to parse.
// prefix - The key to search for (e.g. 'foo:')
// delimiter - The character that separates each config option.
// fullName - The human-readable name of the config option.
// min - The minimum acceptable value.
// max - The maximum acceptable value.
// result - A pointer to where the result should be stored.
// errMsg - The string where error messages should be written.
//
// Returns 1 if the config value was found. 0 if not found. -1 on error.
template <class T> int parseOption(const String config, const String prefix, const char* delimiter, const char* fullName, const T min, const T max, T* result, String& errMsg)
{
    int begin, end;
    int res = parseOptionIndexes(config, prefix, delimiter, &begin, &end);
    if (res != 1)
        return res;
    
    *result = (T)config.substring(begin, end).toFloat();
    
    if (*result < min || *result > max)
    {
        errMsg = String::format("Config error: %s value out-of-range! Valid values: %.4f -> %.4f.", fullName, (float)min, (float)max);
        return -1;
    }
    
    return 1;
}


// Parse out a config value as an array of values.
//
// config - The full input config string to parse.
// prefix - The key to search for (e.g. 'foo:')
// delimiter - The character that separates each config option.
// fullName - The human-readable name of the config option.
// min - The minimum acceptable value.
// max - The maximum acceptable value.
// result - A pointer to the array where the result should be stored.
// len - The number of elements to expect.
// errMsg - The string where error messages should be written.
//
// Returns 1 if the config value was found. 0 if not found. -1 on error.
template <class T> int parseOptionArray(String config, const String prefix, const char* delimiter, const char* fullName, const T min, const T max, T* result, const size_t len, String& errMsg)
{
    int begin, end;
    int res = parseOptionIndexes(config, prefix, String("]") + delimiter, &begin, &end);
    if (res != 1)
        return res;

    String arrayStr = config.substring(begin, end);
    
    // Do basic validation
    if (!arrayStr.startsWith("["))
    {
        errMsg = String::format("Config error: Malformed %s array. Opening bracket missing!", fullName);
        return -1;
    }
    arrayStr.remove(0,1);
    
    // Parse array values
    begin = 0;
    end = 0;
    size_t idx = 0;
    while (idx < len)
    {
        end = arrayStr.indexOf(delimiter, begin);
            
        result[idx] = (T)arrayStr.substring(begin, end).toInt();

        if (result[idx] < min || result[idx] > max)
        {
            errMsg = String::format("Config error: %s value out-of-range! Valid values: %3.4f-%3.4f.", fullName, (float)min, (float)max);
            return -1;
        }
        
        if (end == -1)
            break;
        
        ++idx;
        begin = ++end;
    }
    
    return 1;
}




/**************************************
 * Cloud Object
 **************************************/
Cloud::Cloud() :
    _registered(false),
    _config(0),
    _state(0),
    _prevStateVersion(0),
    _resetTimer(SYSTEM_RESET_TIMEOUT_MS)
{
}
    
void Cloud::setup(Config* config, State* state)
{
    _config = config;
    _state = state;
    _prevStateVersion = -1; // Force initial update of state
    _resetTimer.disable();
    
    // Init cloud variables
    _updateState();
    _onSettingsCalled("{}");
    
    _verifyCloudRegistration();
    
    ThingSpeak.begin(_tcpClient);
}


void Cloud::loop()
{
    if (_resetTimer.enabled() && _resetTimer.expired())
        System.reset();

    _updateState();
    
    _verifyCloudRegistration();
}


void Cloud::publishState()
{
    const Temperature::format_t format = _state->getFormat();
    
    ThingSpeak.setField(1, _state->getAutoSetpoint().get(format));
    ThingSpeak.setField(2, _state->getProbeTemp(0).get(format));
    ThingSpeak.setField(3, _state->getOutputState());
    ThingSpeak.setField(4, _state->getPIDOutput());
    
    float tolerance = _config->getTolerance().get();
    if (format == Temperature::FAHRENHEIT)
        tolerance = _config->getTolerance().get(format) - 32.0;
    ThingSpeak.setField(5, tolerance);
    
    if (_state->getAverageTemp().isValid())
        ThingSpeak.setField(6, _state->getAverageTemp().get(format));

    ThingSpeak.writeFields(TS_CHANNEL_NUMBER, TS_API_WRITE_KEY);
}


void Cloud::_verifyCloudRegistration()
{
    if (_registered)
        return;
        
     if (!Particle.connected())
     {
        _registered = false;
        return;
     }

    if (!Particle.variable("state", _cloudState))
        Serial.println("Error: Failed to register 'state' cloud variable!");
        
    if (!Particle.variable("settings", _cloudSettings))
        Serial.println("Error: Failed to register 'settings' cloud variable!");
        
    if (!Particle.variable("error", _cloudError))
        Serial.println("Error: Failed to register 'error' cloud variable!");
        
    if (!Particle.function("settings", &Cloud::_onSettingsCalled, this))
        Serial.println("Error: Failed to register 'settings' cloud function!");
        
    if (!Particle.function("reset", &Cloud::_onResetCalled, this))
        Serial.println("Error: Failed to register 'reset' cloud function!");

    _registered = true;
}


// {"t":["<float|-->","<float|-->"],"s":<int>}
void Cloud::_updateState()
{
    if (_state->version() == _prevStateVersion)
        return;
    
    _cloudState = "{\"t\":[";
    for (size_t ii=0; ii<2; ++ii)
    {
        if (ii>0)
            _cloudState += ",";

        if (_state->getProbeTemp(ii).isValid())
            _cloudState += String::format("\"%d\"", _state->getProbeTemp(ii).getRaw());
        else
            _cloudState += "\"--\"";
    }
    _cloudState += String::format("],\"s\":%d}", _state->getOutputState());
    
    _prevStateVersion = _state->version();
}


// config - {sp:<celsius>,tl:<celsius>,asc:<ms>,o:[<celsius>,<celsius>],led:<int>,h:<0|1>,c:<0|1>,kp:<float>,ki:<float>,kd:<float>,aa:<0|1>,asp:<celsius>,atp:<hours>}
int Cloud::_onSettingsCalled(String config)
{
    // Prepare string for parsing
    config.trim();
    if (!config.startsWith("{"))
    {
        _cloudError = "Config error: Malformed JSON object. Opening brace missing!";
        return -1;
    }
    if (!config.endsWith("}"))
    {
        _cloudError = "Config error: Malformed JSON object. Closing brace missing!";
        return -1;
    }

    // Target temp
    {
        celsius_t value;
        int res = parseOption<celsius_t>(config, "sp:", ",", "Target temperature", MIN_TEMP, MAX_TEMP, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setSetpoint(value);
    }
    
    // Tolerance
    {
        celsius_t value;
        int res = parseOption<celsius_t>(config, "tl:", ",", "Tolerance", MIN_TOLERANCE, MAX_TOLERANCE, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setTolerance(value);
    }
    
    // ASC
    {
        unsigned long value = 0;
        int res = parseOption<unsigned long>(config, "asc:", ",", "Anti-short-cycle", ASC_MIN_MS, ASC_MAX_MS, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setAsc(value);
    }
    
    // Probe offsets
    {
        celsius_t values[2] = {0};
        int res = parseOptionArray<celsius_t>(config, "o:", ",", "Probe offsets", OFFSET_MIN, OFFSET_MAX, values, 2, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
        {
            for (size_t ii=0; ii<2; ++ii)
                _config->setProbeOffset(ii, values[ii]);
        }
    }
    
    // LED level
    {
        unsigned char value = 0;
        int res = parseOption<unsigned char>(config, "led:", ",", "LED level", 0, 100, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setLEDLevel(value);
    }
    
    // Heat feature enabled
    {
        int value = 0;
        int res = parseOption<int>(config, "h:", ",", "Heating enabled", 0, 1, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setHeatEnabled(value==1);
    }
    
    // Cool feature enabled
    {
        int value = 0;
        int res = parseOption<int>(config, "c:", ",", "Cooling enabled", 0, 1, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setCoolEnabled(value==1);
    }
    
    // Kp
    {
        float value = 0;
        int res = parseOption<float>(config, "kp:", ",", "PID Kp", PID_COEFF_MIN, PID_COEFF_MAX, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setPidKp(value);
    }
    
    // Ki
    {
        float value = 0;
        int res = parseOption<float>(config, "ki:", ",", "PID Ki", PID_COEFF_MIN, PID_COEFF_MAX, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setPidKi(value);
    }
    
    // Kd
    {
        float value = 0;
        int res = parseOption<float>(config, "kd:", ",", "PID Kd", PID_COEFF_MIN, PID_COEFF_MAX, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setPidKd(value);
    }
    
    // Auto-adjust feature enabled
    {
        int value = 0;
        int res = parseOption<int>(config, "aa:", ",", "Auto-adjust enabled", 0, 1, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
        {
            // Set the starting timestamp if we are enabling auto-adjusting.
            if (!_config->getAutoAdjustEnabled() && value==1)
                _config->setAutoAdjustStartTS( Time.now() );

            _config->setAutoAdjustEnabled(value==1);
        }
    }
    
    // Auto-adjust target temp
    {
        celsius_t value;
        int res = parseOption<celsius_t>(config, "asp:", ",", "Auto-adjust Target temperature", MIN_TEMP, MAX_TEMP, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setAutoSetpoint(value);
    }
    
    // Auto-adjust time period (seconds)
    {
        unsigned long value = 0;
        int res = parseOption<unsigned long>(config, "atp:", ",", "Auto-adjust time period", AUTO_ADJUST_TIME_MIN, AUTO_ADJUST_TIME_MAX, &value, _cloudError);
        if (res == -1)
            return -1;
        else if (res == 1)
            _config->setAutoTimePeriod( value==0 ? 1 : value ); // don't allow a zero value
    }
    
    _config->store();
    
    // Build 'settings' cloud variable string.
    _cloudSettings = String::format("{\"sp\":%d,\"tl\":%d,\"asc\":%d,\"o\":[",
        _config->getSetpoint().getRaw(), 
        _config->getTolerance().getRaw(),
        _config->getAsc());
    for (size_t ii=0; ii<2; ++ii)
    {
        if (ii != 0)
            _cloudSettings += ",";
        _cloudSettings += String(_config->getProbeOffset(ii).getRaw());
    }
    _cloudSettings += String::format("],\"led\":%d,\"h\":%d,\"c\":%d,\"kp\":%.4f,\"ki\":%.4f,\"kd\":%.4f,", 
        _config->getLEDLevel(), _config->getHeatEnabled()?1:0, _config->getCoolEnabled()?1:0, _config->getPidKp(), _config->getPidKi(), _config->getPidKd());
        
    _cloudSettings += String::format("\"aa\":%d,\"asp\":%d,\"atp\":%d}",
        _config->getAutoAdjustEnabled()?1:0, _config->getAutoSetpoint().getRaw(), _config->getAutoTimePeriod());

    return 0;
}



int Cloud::_onResetCalled(String ignored)
{
    _resetTimer.restart();
    return 0;
}

