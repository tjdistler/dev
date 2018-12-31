#include "Common.h"
#include "Config.h"
#include "State.h"
#include "Controller.h"
#include "Cloud.h"
#include "Hardware.h"
#include "UI.h"

// Included to stop annoying IDE warning
#include "ThermalProbe.h"
#include "RotaryEncoder.h"

SYSTEM_THREAD(ENABLED);
STARTUP(WiFi.selectAntenna(ANT_EXTERNAL));

#define VERSION_MAJOR   0
#define VERSION_MINOR   8
#define VERSION_BUILD   6


/**************************************
 * Variables
 **************************************/
Config config;
State state( State::ALL_OFF );
Controller controller;
Cloud cloud;
Hardware hw;
UI ui( UI_UPDATE_INTERVAL_MS );

unsigned long configVersion = -1; // Version of the config we are running with.
bool updating = false; // Indicates if update in progress

PolledTimer readTimer(TEMPERATURE_READ_INTERVAL_MS);
int probeErrorCounts[2] = {0};


/**************************************
 * Functions
 **************************************/
void readTemperature();
void heatOn();
void heatOff();
void coolOn();
void coolOff();
void updateSettings();
void onFwUpdate(system_event_t event, int param, void* data);



bool logToCloud(const char* message)
{
    char msg[256];
    strncpy(msg, message, 255);
    msg[255] = '\0';

    return Particle.publish("ctrl-log", msg, 60, PRIVATE);
}
 

void setup()
{
    System.on(firmware_update, onFwUpdate);
    
    hw.setup();
    
    config.load();
    
    state.setFormat(Temperature::FAHRENHEIT);
    
    controller.setup(&config, &state);
    
    cloud.setup(&config, &state);
    ui.setup(&hw, &config, &state);
    
    readTemperature();
    
    readTimer.restart();

    Serial.println(String::format("Firmware %d.%d.%d running.", VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD));
}


void loop()
{
    hw.loop();
    ui.loop();
    
    if (updating)
        return;
        
    cloud.loop();
    
    updateSettings();
    
    if (readTimer.expired())
    {
        readTemperature();

        if (!state.probeTempReady(0))
        {
            heatOff();
            coolOff();
        }
        else
        {
            Controller::decision_t decision = controller.compute( state.getProbeTemp(0) );
            
            if (decision == Controller::HEAT_ASC)
            {
                Serial.println("State: HEAT_ASC");
                heatOff();
                coolOff();
                state.setOutputState(State::HEAT_ASC);
            }
            else if (decision == Controller::HEAT)
            {
                Serial.println("State: HEAT");
                coolOff();
                heatOn();
                state.setOutputState(State::HEAT_ON);
            }
            else if (decision == Controller::COOL_ASC)
            {
                Serial.println("State: COOL_ASC");
                heatOff();
                coolOff();
                state.setOutputState(State::COOL_ASC);
            }
            else if (decision == Controller::COOL)
            {
                Serial.println("State: COOL");
                heatOff();
                coolOn();
                state.setOutputState(State::COOL_ON);
            }
            else
            {
                Serial.println("State: OFF");
                heatOff();
                coolOff();
                state.setOutputState(State::ALL_OFF);
            }
            
            cloud.publishState();
        }
    }
}


void readTemperature()
{
    for (size_t ii=0; ii<2; ++ii)
    {
        ThermalProbe& probe = hw.probe(ii);
        bool error;
        if (probe.complete(error))
        {
            Temperature temp = probe.getTemperature();
            if (!error)
            {
                if (temp.isValid())
                {
                    state.setProbeTemp(ii, temp + config.getProbeOffset(ii)); // Apply temperature offset.
                    probeErrorCounts[ii] = 0;
                }
                else
                {
                    // startup state of ThermalProbe class. Do nothing.
                }
            }
            else if (++(probeErrorCounts[ii]) >= NUM_PROBE_ERRORS_ALLOWED)
            {
                // Too many successive read errors.
                state.setProbeTemp(ii, Temperature::UNDEFINED);
                probeErrorCounts[ii] = 0;
            }
        }
        
        // Initiate next read
        probe.readBegin();
    }
}


void heatOn()
{
    hw.relayOpen(Hardware::COOL_RELAY);
    hw.relayClose(Hardware::HEAT_RELAY);
}

void heatOff()
{
    hw.relayOpen(Hardware::COOL_RELAY);
    hw.relayOpen(Hardware::HEAT_RELAY);
}

void coolOn()
{
    hw.relayOpen(Hardware::HEAT_RELAY);
    hw.relayClose(Hardware::COOL_RELAY);
}

void coolOff()
{
    hw.relayOpen(Hardware::HEAT_RELAY);
    hw.relayOpen(Hardware::COOL_RELAY);
}


void updateSettings()
{
    if (config.version() != configVersion)
    {
        const float level = config.getLEDLevel() / 100.0;
        hw.led(Hardware::RED_LED).setLevel(level);
        hw.led(Hardware::BLUE_LED).setLevel(level);
        hw.led(Hardware::STATUS_LED).setLevel(level);
        
        configVersion = config.version();
    }
}


void onFwUpdate(system_event_t event, int param, void* data)
{
    switch(param)
    {
    case firmware_update_begin:
    case firmware_update_progress:
        updating = true;
        heatOff();
        coolOff();
        break;
    case firmware_update_complete:
    case firmware_update_failed:
        System.reset();
        break;
    };
    
    ui.onFirmwareUpdate(event, param, data);
}