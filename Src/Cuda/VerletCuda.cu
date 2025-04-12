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
__global__ void UpdatePositionsKernel(CudaWorld* cuda_world) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Array<VerletParticle> particles = cuda_world->particles;
    if (idx >= particles.size || particles.data[idx].is_static) 
        return;

    VerletParticle* p = &particles.data[idx];
    
    // Store current position
    float2 temp = p->position;
    
    // Verlet integration (no acceleration, just inertia)
    p->position = p->position + (p->position - p->old_position);
    p->old_position = temp;
}

__global__ void UpdateConstraintsKernel(CudaWorld* cuda_world) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Array<VerletConstraint> constraints = cuda_world->constraints;
    if (idx >= constraints.size)
        return;

    VerletConstraint* c = &constraints.data[idx];
    VerletParticle* p1 = c->particle1;
    VerletParticle* p2 = c->particle2;

    float2 delta = p2->position - p1->position;
    float dist = Length(delta);
    float error = (dist - c->rest_length) / dist;
    float2 correction = delta * error * c->stiffness;

    p1->position += correction;
    p2->position -= correction;
}

void UpdateVerletParticles(CudaWorld* cuda_world) 
{
    const int block_size = 256;
    const int num_blocks = (cuda_world->particles.size + block_size - 1) / block_size;
    
    // Update positions with fixed timestep
    UpdatePositionsKernel<<<num_blocks, block_size>>>(cuda_world);
    cudaDeviceSynchronize();

    const int constraint_block_size = 256;
    const int num_constraint_blocks = (cuda_world->constraints.size + constraint_block_size - 1) / constraint_block_size;
    UpdateConstraintsKernel<<<num_constraint_blocks, constraint_block_size>>>(cuda_world);
    cudaDeviceSynchronize();
    
    // Solve collisions
    // SolveCollisionsKernel<<<num_blocks, block_size>>>(particles, num_particles);
    // cudaDeviceSynchronize();
}
