#pragma once
#ifdef _WIN32
#include <windows.h>
#else
#include       <sys/time.h>
#include       <time.h>
#endif
#include <string>

extern double time_gelu;
extern double time_ln;
extern double time_ffn;
extern double time_attn;
extern double time_attn_mul;
extern double time_conv1d;
extern double time_output;


namespace util{ \

class timer_c 
{
public:
    timer_c();
    ~timer_c();
    void m_start_timer();
    void m_end_timer();
    double m_get_time_diff_sec();
    double m_get_time_diff_msec();

public:
    static void m_current_time_to_str(std::string & output);

private:
    /* get the time difference in unit of usec(1.0e-6 sec) */
    double m_get_time_diff();

private:
#ifdef  _WIN32
    DWORD  m_start_time;
    DWORD  m_end_time;
#else
    struct timeval             m_start_time;
    struct timeval             m_end_time;
#endif

};

}
