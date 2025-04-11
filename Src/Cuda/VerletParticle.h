#pragma once

#include <cuda_runtime.h>

struct VerletParticle {
    float2 position;      // Current position
    float2 old_position;  // Previous position
    float radius;         // Particle radius
    int is_static;        // 0 = movable, 1 = static (immovable)
};
