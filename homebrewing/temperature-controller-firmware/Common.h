#ifndef _COMMON_H_
#define _COMMON_H_

#include "application.h"
#include <inttypes.h>

/**************************************
 * Base Temperature Datatype
 **************************************/
typedef long celsius_t;   // Represented as 1/10,000th of a degree (i.e. 22.3C = 223000)

const float CELSIUS_T_SCALE_FACTOR = 10000.0f;

const char DEGREE_CHAR = 0xdf;

#define CELSIUS_T_TO_CELSIUS(v)     ((float)((v) / CELSIUS_T_SCALE_FACTOR))
#define CELSIUS_TO_CELSIUS_T(v)     ((celsius_t)((v) * CELSIUS_T_SCALE_FACTOR))
#define CELSIUS_T_TO_FAHRENHEIT(v)  (CELSIUS_T_TO_CELSIUS(v) * 1.8 + 32.0)
#define FAHRENHEIT_TO_CELSIUS_T(v)  (CELSIUS_TO_CELSIUS_T( ((v) - 32.0) / 1.8 ))


/**************************************
 * Default Controller Constants
 **************************************/
const celsius_t     DEFAULT_SETPOINT         = FAHRENHEIT_TO_CELSIUS_T(70);
const celsius_t     DEFAULT_TOLERANCE        = 5556;  // 1F
const unsigned long DEFAULT_ASC              = 15 * 60 * 1000;
const float         DEFAULT_KP               = 0.75;
const float         DEFAULT_KI               = 1000;
const float         DEFAULT_KD               = 0.0;
const unsigned long DEFAULT_AUTO_TIME_PERIOD = 24 * 3600; // seconds

const unsigned long TEMPERATURE_READ_INTERVAL_MS    = 15 * 1000;
const int           NUM_PROBE_ERRORS_ALLOWED        = 5;
const size_t        PROBE_WINDOW_SIZE               = (60000 / TEMPERATURE_READ_INTERVAL_MS) * 3; // 3 mins

const unsigned long TEMP_AVERAGE_WINDOW_SIZE_M      = 12 * 60; // 12hrs
const size_t        TEMP_AVERAGE_WINDOW_SAMPLES     = TEMP_AVERAGE_WINDOW_SIZE_M * (60000 / TEMPERATURE_READ_INTERVAL_MS);

const unsigned long PID_DERIVATIVE_WINDOW_M         = 5; // 5 mins
const size_t        PID_DERIVATIVE_WINDOW_SIZE      = PID_DERIVATIVE_WINDOW_M * (60000 / TEMPERATURE_READ_INTERVAL_MS);

const float         TOLERANCE_PERCENTAGE_LIMIT      = 0.90f;

const unsigned long UI_UPDATE_INTERVAL_MS           = 250;
const unsigned long UI_BACKLIGHT_ON_MS              = 120 * 1000;
const unsigned long UI_NO_WIFI_LED_BLINK_MS         = 850;
const unsigned long UI_NO_INTERNET_LED_BLINK_MS     = 250;
const unsigned long UI_ASC_BLINK_MS                 = 1000;
const unsigned long UI_FIRWARE_UPDATE_BLINK_MS      = 500;

// Thingspeak
const unsigned long TS_CHANNEL_NUMBER   = 210613;
const String        TS_API_WRITE_KEY    = "D2RLJFRW7TK6ET3K";

const unsigned long SYSTEM_RESET_TIMEOUT_MS = 1000; // Delays system reset after reset function called.


/**************************************
 * Range Constants
 **************************************/
const celsius_t MAX_TEMP = CELSIUS_TO_CELSIUS_T(125); // As defined by DS18B20 data sheet.
const celsius_t MIN_TEMP = CELSIUS_TO_CELSIUS_T(-55);
const celsius_t MAX_TOLERANCE = CELSIUS_TO_CELSIUS_T(25);
const celsius_t MIN_TOLERANCE = CELSIUS_TO_CELSIUS_T(0);
const int ASC_MAX_MS = 60 * 60 * 1000;
const int ASC_MIN_MS = 0;
const celsius_t OFFSET_MAX = CELSIUS_TO_CELSIUS_T(10);
const celsius_t OFFSET_MIN = CELSIUS_TO_CELSIUS_T(-10);
const float PID_COEFF_MAX = 10000.0f;
const float PID_COEFF_MIN = 0.0f;
const unsigned long AUTO_ADJUST_TIME_MIN = 1;
const unsigned long AUTO_ADJUST_TIME_MAX = 365*24*60*60; // one year


/**************************************
 * Hardware Constants
 **************************************/
const int BLUE_LED_PIN = WKP;
const int RED_LED_PIN = RX;
const int STATUS_LED_PIN = TX;
const int DS_BUS_0_PIN = D2;
const int DS_BUS_1_PIN = D3;
const int RELAY_0_PIN = D5;
const int RELAY_1_PIN = D6;
const int ENCODER_BTN_PIN = D4;
const int ENCODER_A_PIN = A0;
const int ENCODER_B_PIN = A1;
const int IR_EMITTER_PIN = A5;
const int IR_DETECTOR_PIN = A4;

const bool DS_BUS_0_PARASITIC = true;
const bool DS_BUS_1_PARASITIC = true;

const int SERIAL_BAUD = 9600;

const int LED_ON_STATE = LOW;
const int LED_PWM_HZ = 500;

const int RELAY_CLOSE_STATE = LOW;

const int LCD_ADDR = 0x27;
const int LCD_COLUMNS = 20;
const int LCD_ROWS = 4;

const unsigned long ENCODER_READ_INTERVAL_MS = 5;
const unsigned long ENCODER_ACTIVE_READ_DURATION_MS = 1000; // Only allow the system timer for the encoder to run this long in response to interrupt.
const unsigned long ENCODER_FAST_ROTATION_MIN_MS = 100;     // If multiple ticks of the encoder happen w/in this time, then mark as "fast"
const int ENCODER_FAST_ROTATION_TICKS = 16;                 // Number of required "fast" ticks before we scale.
const int ENCODER_FAST_SCALAR = 10;
const int ENCODER_BUTTON_DEBOUNCE_COUNT = 3;
const int ENCODER_BUTTON_PRESSED_STATE = LOW;
const int ENCODER_BUTTON_RELEASED_STATE = HIGH;


/**************************************
 * LCD Constants
 **************************************/
const uint8_t WIFI_CHAR[4][8] = {
{
    0b00000,
    0b00001,
    0b00011,
    0b00111,
    0b01111,
    0b01111,
    0b00000,
    0
},
{
    0b00000,
    0b00000,
    0b00010,
    0b00110,
    0b01110,
    0b01110,
    0b00000,
    0
},
{
    0b00000,
    0b00000,
    0b00000,
    0b00100,
    0b01100,
    0b01100,
    0b00000,
    0
},
{
    0b00000,
    0b00000,
    0b00000,
    0b00000,
    0b01000,
    0b01000,
    0b00000,
    0
}
};
    
const uint8_t CHECK_CHAR[8] = {
    0b00000,
    0b00000,
    0b00001,
    0b00010,
    0b10100,
    0b01000,
    0b00000,
    0
};

    
const uint8_t WIFI_CHAR_ADDR[4] = {0, 1, 2, 3};
const uint8_t CHECK_CHAR_ADDR = 4;



/**************************************
 * Helper Functions
 **************************************/
template<class T>
T range(const T value, const T high, const T low) {
    T result = value;
    if (value > high)
        result = high;
    else if (value < low)
        result = low;
    return result;
}
 
 
extern bool logToCloud(const char* message);



/**************************************
 * Temperature Object
 **************************************/
class Temperature
{
public:

    typedef enum {
        CELSIUS,
        FAHRENHEIT
    } format_t;
    
    static const celsius_t UNDEFINED = -999999;

    Temperature(const celsius_t value=UNDEFINED) : _raw(value) {}
    
    Temperature(const Temperature& other) {
        operator=(other);
    }
    Temperature& operator=(const Temperature& other) {
        _raw = other._raw;
        return *this;
    }
    Temperature& operator=(const celsius_t& value) {
        _raw = value;
        return *this;
    }
    
    bool isValid() const { return _raw != UNDEFINED; }
    
    void setRaw(const celsius_t value) { _raw = value; }
    celsius_t getRaw() const { return _raw; };

    void set(const float value, const format_t format=CELSIUS) {
        if (format == CELSIUS)
            _raw = CELSIUS_TO_CELSIUS_T(value);
        else
            _raw = FAHRENHEIT_TO_CELSIUS_T(value);
    }
    
    float get(const format_t format=CELSIUS) const {
        if (format == CELSIUS)
            return CELSIUS_T_TO_CELSIUS(_raw);
        return CELSIUS_T_TO_FAHRENHEIT(_raw);
    }
    
    Temperature operator+(const Temperature& rhs) const {
        Temperature result = *this;
        result._raw += rhs._raw;
        return result;
    }
    
    Temperature operator-(const Temperature& rhs) const {
        Temperature result = *this;
        result._raw -= rhs._raw;
        return result;
    }
    
    Temperature operator*(const double rhs) const {
        Temperature result = *this;
        result._raw *= rhs;
        return result;
    }
    
    Temperature operator/(const Temperature& rhs) const {
        Temperature result = *this;
        result._raw /= rhs._raw;
        return result;
    }
    
    Temperature& operator+=(const Temperature& rhs) {
        this->_raw += rhs._raw;
        return *this;
    }
    
    bool operator==(const Temperature& rhs) {
        return _raw == rhs._raw;
    }
    
    bool operator!=(const Temperature& rhs) {
        return !(*this == rhs);
    }
    
private:
    celsius_t _raw;
};



/**************************************
 * PolledTimer Object
 **************************************/
class PolledTimer
{
public:
    PolledTimer(unsigned long intervalMS=0) :
        _interval(intervalMS),
        _prev(millis()),
        _enabled(intervalMS != 0)
    {}

  
    void restart() { _prev = millis(); _enabled = true; }
    
    bool expired(bool autoReset=true) {
        if (!_enabled || _interval == 0)
            return false;
        
        unsigned long now = millis();
        unsigned long delta = now - _prev;
        if (delta >= _interval) {
            if (autoReset)
                _prev = now - (delta - _interval);
            return true;
        }
        return false;
    }

    void setInterval(unsigned long intervalMS) { _interval = intervalMS; }
    unsigned long getInterval() const { return _interval; }
    void disable() { _enabled = false; }
    void enable() { _enabled = true; }
    bool disabled() const { return !_enabled; }
    bool enabled() const { return _enabled; }
  
private:
    unsigned long _interval; // ms
    unsigned long _prev;     // The last time the timer expired
    bool _enabled;
};



/**************************************
 * Debounce object
 * 
 * Only returns a value if that value has been consistently read
 * a certain number of times.
 **************************************/
template<class T, size_t Size>
class Debounce
{
public:
    Debounce() : _next(0), _count(0) {}
    
    bool ready() const
    {
        if (_count < Size)
            return false;
        
        T first = _data[0];
        for (size_t ii=1; ii<Size; ++ii)
        {
            if (_data[ii] != first)
                return false;
        }
        
        return true;
    }
    
    T get() const { return _data[0]; }
    
    void add(const T sample)
    {
        _data[_next] = sample;
        _next = (_next+1) % Size;
        if (++_count > Size)
            _count = Size;
    }
    
    void reset()
    {
        _next = 0;
        _count = 0;
    }
    
private:
    size_t _next;
    size_t _count;
    T _data[Size];
};




/**************************************
 * DataSet
 **************************************/
template<class T, size_t windowSize>
class DataSet
{
public:
    DataSet() : _ready(false), _next(0), _count(0) {}
    ~DataSet() {}
    
    bool ready() const { return _ready; }
    
    size_t count() const { return _count; }
    
    void reset() {
        _ready = false;
        _next = 0;
        _count = 0;
    }
    
    void add(const T value) {
        _data[_next] = value; // Replace oldest value w/ newest.
        _next = (_next+1) % windowSize;
        _count = (_count >= windowSize) ? windowSize : (_count + 1);
        _ready = true;
    }
    
    T oldest() const {
        if (_count >= windowSize)
            return _data[_next];
        return _data[0];
    }
    
    T newest() const {
        if (_next > 0)
            return _data[_next-1];
        return _data[windowSize-1];
    }
    
    T get(const size_t index) const { return _data[index]; }
    
private:
    bool _ready;
    T _data[windowSize];
    size_t _next;
    size_t _count;
};




/**************************************
 * Simple Moving Average
 **************************************/
template<class T, size_t windowSize>
class SimpleMovingAverage : public DataSet<T, windowSize>
{
    typedef DataSet<T, windowSize> Base;
    
public:
    SimpleMovingAverage() : Base() {}
    ~SimpleMovingAverage() {}
    
    bool ready() const { return Base::ready(); }
    
    void reset(const T value=0) {
        Base::reset();
        _curSMA = value;
    }
    
    // cur = prev + p(M)/n - p(M-n)/n
    void add(const T value) {
        
        if (Base::count() < windowSize)
        {
            // Calculate the full average.
            T sum = 0;
            for (size_t ii=0; ii<Base::count(); ++ii)
                sum += Base::get(ii);
            sum += value;
            _curSMA = sum / (Base::count() + 1);
        }
        else
        {
            // Calculate the moving average.
            _curSMA = _curSMA + ( (value / windowSize) - (Base::oldest() / windowSize) );
        }
        
        Base::add(value);
    }
    
    T get() const { return _curSMA; }
    
private:
    T _curSMA;
};



/**************************************
 * Exponential Moving Average
 **************************************/
template<class T>
class ExponentialMovingAverage
{
public:
    // alpha - [0.0, 1.0] - Degree of weighted decrease. Higher values discount older samples faster.
    ExponentialMovingAverage(const float alpha) : _alpha(alpha), _initialized(false) {}
    ~ExponentialMovingAverage() {}
    
    void reset() {
        _initialized = false;
    }
    
    // St = a*Yt + (1-a)*St-1
    void add(const T value) {
        if (!_initialized) {
            _curEMA = value;
            _initialized = true;
            return;
        }
        
        _curEMA = (value*_alpha) + (_curEMA * (1.0-_alpha));
    }
    
    bool ready() const { return _initialized; }
    
    T get() const { return _curEMA; }
    
private:
    float _alpha;
    bool _initialized;
    T _curEMA;  // St-1 
};


#endif