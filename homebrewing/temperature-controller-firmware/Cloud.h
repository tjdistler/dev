#ifndef _CLOUD_H_
#define _CLOUD_H_

#include "Common.h"
#include "Config.h"
#include "State.h"

class Cloud
{
public:
    Cloud();
    
    void setup(Config* config, State* state);
    
    void loop();
    
    void publishState();
    
private:
    bool _registered;   // Indicates if all variable have been registered w/ the cloud.
    String _cloudState;
    String _cloudSettings;
    String _cloudError; // String descripting last error that occurred
    Config* _config;
    State* _state;
    unsigned long _prevStateVersion;
    PolledTimer _resetTimer; // Gives the API a chance to return before resetting the system.
    
    TCPClient _tcpClient;
    
    void _verifyCloudRegistration();
    
    void _updateState();
    
    // Callback to handle cloud function 'settings' call.
    int _onSettingsCalled(String config);
    
    // Callback to handle cloud function 'reset' call.
    int _onResetCalled(String ignored);
};

#endif