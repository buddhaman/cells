#pragma once

#include "AnymUtil.h"

// Basic circular buffer for timing samples
template <typename T>
struct CircularBuffer
{
    T* data;
    I32 capacity;
    I32 current;
    
    CircularBuffer(I32 size) : capacity(size), current(0)
    {
        data = (T*)malloc(sizeof(T) * size);
        memset(data, 0, sizeof(T) * size);
    }
    
    ~CircularBuffer()
    {
        free(data);
    }
    
    void Add(T value)
    {
        data[current] = value;
        current = (current + 1) % capacity;
    }
    
    T GetAverage() const
    {
        T sum = 0;
        for (I32 i = 0; i < capacity; i++)
            sum += data[i];
        return sum / capacity;
    }
    
    T GetMax() const
    {
        T max = data[0];
        for (I32 i = 1; i < capacity; i++)
            if (data[i] > max)
                max = data[i];
        return max;
    }
    
    T GetMin() const
    {
        T min = data[0];
        for (I32 i = 1; i < capacity; i++)
            if (data[i] < min && data[i] > 0) // Ignore zero entries
                min = data[i];
        return min;
    }
};