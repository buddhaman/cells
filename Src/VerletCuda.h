#pragma once

#include <cuda_runtime.h>

struct VerletParticle {
    float2 position;
    float2 old_position;
    float2 acceleration;
};

void VerletIntegration(VerletParticle* particles, int num_particles, float dt); 