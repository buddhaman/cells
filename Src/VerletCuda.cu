#include "VerletCuda.h"
#include <device_launch_parameters.h>

// Helper function for vector operations
__device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

__device__ float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

__device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__global__ void VerletIntegrationKernel(VerletParticle* particles, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    VerletParticle& p = particles[idx];
    
    // Store current position
    float2 temp = p.position;
    
    // Verlet integration
    p.position = 2.0f * p.position - p.old_position + p.acceleration * (dt * dt);
    
    // Update old position
    p.old_position = temp;
}

// Host-side wrapper function
void VerletIntegration(VerletParticle* particles, int num_particles, float dt) {
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;
    
    VerletIntegrationKernel<<<numBlocks, blockSize>>>(particles, num_particles, dt);
    cudaDeviceSynchronize();
} 