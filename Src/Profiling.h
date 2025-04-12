#pragma once

#include <string>
#include <unordered_map>

// Include AnymUtil.h first to avoid macro redefinition
#include "AnymUtil.h"
#include "CircularBuffer.h"

// Maximum no extra windows stuff
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

constexpr I32 PROFILER_SAMPLE_COUNT = 120;

// Performance metrics for a section
struct ProfilerMetric
{
    CircularBuffer<U64> samples;
    const char* name;
    
    ProfilerMetric(const char* name, I32 sample_count) 
        : samples(sample_count), name(name) {}
};

struct ProfileScope
{
    static std::unordered_map<const char*, ProfilerMetric*> map;
    
    static ProfilerMetric* RegisterProfilerMetric(const char* name)
    {
        auto it = map.find(name);
        if (it == map.end()) {
            map[name] = new ProfilerMetric(name, PROFILER_SAMPLE_COUNT);
        }
        return map[name];
    }

    static void ResetMetrics()
    {
        for (auto& pair : map) {
            memset(pair.second->samples.data, 0, pair.second->samples.capacity * sizeof(U64));
        }
    }

    U64 start;
    ProfilerMetric* metric;

    ProfileScope(ProfilerMetric* metric_) : metric(metric_)
    {
        start = GetTickCount64();
    }

    ~ProfileScope()
    {
        U64 end = GetTickCount64();
        U64 duration = end - start;
        metric->samples.Add(duration);
    }
};

// Initialize static map
std::unordered_map<const char*, ProfilerMetric*> ProfileScope::map;

// Helper macros
#define PROFILE_SCOPE(name) \
    static ProfilerMetric* PROFILER_METRIC_##__LINE__ = \
        ProfileScope::RegisterProfilerMetric(name); \
    ProfileScope profiler_scope_##__LINE__(PROFILER_METRIC_##__LINE__)

#define PROFILE_RESET() ProfileScope::ResetMetrics()
