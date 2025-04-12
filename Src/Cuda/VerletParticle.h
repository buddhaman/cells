#pragma once

#include <cuda_runtime.h>

struct VerletParticle 
{
    float2 position;      
    float2 old_position; 
    float radius;       
    float mass;        
    int is_static;        // 0 = movable, 1 = static (immovable)
};
