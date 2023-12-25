#include "timer.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

double time_gelu=0;
double time_ln=0;
double time_ffn=0;
double time_attn=0;
double time_attn_mul=0;
double time_conv1d=0;
double time_output=0;

namespace util{ \

timer_c::timer_c() {
}

timer_c::~timer_c() {
}

void timer_c::m_start_timer() {

#ifdef  _WIN32
    m_start_time = GetTickCount();
#else
    gettimeofday(&this->m_start_time, NULL);
#endif

}

void timer_c::m_end_timer() {
#ifdef USE_CUDA
    cudaThreadSynchronize();
#endif

#ifdef  _WIN32
    m_end_time = GetTickCount();
#else
    gettimeofday(&this->m_end_time, NULL);
#endif

}

double timer_c::m_get_time_diff_sec() {

    double time_diff_sec = this->m_get_time_diff() / 1000;
    return time_diff_sec;

}

double timer_c::m_get_time_diff_msec() {

    double time_diff_msec = this->m_get_time_diff();
    return time_diff_msec;

}

// return ms
double timer_c::m_get_time_diff() {

    double time_diff = 0.0;
#ifdef  _WIN32
    time_diff = m_end_time - m_start_time;
#else
    time_diff = (this->m_end_time.tv_sec - this->m_start_time.tv_sec) * 1000 + ((double)(this->m_end_time.tv_usec - this->m_start_time.tv_usec)) / 1000;
#endif
    return time_diff;

}

void timer_c::m_current_time_to_str(std::string & output)
{
    const time_t t = time(NULL);
    struct  tm * current_time = localtime(&t);
    char temp_time[100];
    memset(temp_time, 0, sizeof(char)* 100);
    sprintf(temp_time, "[%04d-%02d-%02d %02d:%02d:%02d]", current_time->tm_year + 1900, current_time->tm_mon + 1, current_time->tm_mday, current_time->tm_hour, current_time->tm_min, current_time->tm_sec);
    output = temp_time;
    return;
}
}