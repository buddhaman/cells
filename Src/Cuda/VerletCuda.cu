#include "VerletCuda.h"
#include <device_launch_parameters.h>

// Vector operators for CUDA float2
__device__ __forceinline__ float2 
operator+(float2 a, float2 b) 
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 
operator-(float2 a, float2 b) 
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 
operator*(float2 a, float b) 
{
    return make_float2(a.x * b, a.y * b);
}

__device__ __forceinline__ float2 
operator*(float b, float2 a) 
{
    return make_float2(a.x * b, a.y * b);
}

__device__ __forceinline__ void 
operator+=(float2& a, float2 b) 
{
    a.x += b.x;
    a.y += b.y;
}

__device__ __forceinline__ void 
operator-=(float2& a, float2 b) 
{
    a.x -= b.x;
    a.y -= b.y;
}

__device__ float 
Length(float2 v) 
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

// Helper functions for vector operations
__device__ float2 normalize(float2 v) {
    float len = Length(v);
    if (len > 0.0001f) {
        return make_float2(v.x / len, v.y / len);
    }
    return make_float2(0.0f, 0.0f);
}

// CUDA kernel for Verlet integration
__global__ void UpdatePositionsKernel(VerletParticle* particles, int num_particles) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || particles[idx].is_static) 
        return;

    VerletParticle* p = &particles[idx];
    
    // Store current position
    float2 temp = p->position;
    
    // Verlet integration (no acceleration, just inertia)
    p->position = p->position + (p->position - p->old_position);
    p->old_position = temp;
}

__global__ void SolveCollisionsKernel(VerletParticle* particles, int num_particles) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || particles[idx].is_static) 
        return;
    
    VerletParticle* p1 = &particles[idx];
    
    for (int i = idx + 1; i < num_particles; i++) 
    {
        VerletParticle* p2 = &particles[i];
        if (p2->is_static) 
            continue;
        
        float2 delta = p2->position - p1->position;
        float dist = Length(delta);
        
        float min_dist = p1->radius + p2->radius;
        if (dist < min_dist && dist > 0.0001f) 
        {
            float correction_factor = (min_dist - dist) / dist * 0.5f;
            float2 correction = delta * correction_factor;
            
            if (!p1->is_static) p1->position -= correction;
            if (!p2->is_static) p2->position += correction;
        }
    }
}

void UpdateVerletParticles(VerletParticle* particles, int num_particles) 
{
    const int block_size = 256;
    const int num_blocks = (num_particles + block_size - 1) / block_size;
    
    // Update positions with fixed timestep
    UpdatePositionsKernel<<<num_blocks, block_size>>>(particles, num_particles);
    cudaDeviceSynchronize();
    
    // Solve collisions
    // SolveCollisionsKernel<<<num_blocks, block_size>>>(particles, num_particles);
    // cudaDeviceSynchronize();
}

// Main simulation step
void VerletSimulationStep(VerletParticle* particles, int num_particles)
{
    UpdateVerletParticles(particles, num_particles);
} 