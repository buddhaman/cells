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
    
    float2 velocity = (p->position - p->old_position) * 0.94f;
    p->position = p->position + velocity;
    p->old_position = temp;
}

// First pass: Calculate constraint corrections
__global__ void CalculateConstraintCorrectionsKernel(CudaWorld* cuda_world) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Array<VerletConstraint> constraints = cuda_world->constraints;
    if (idx >= constraints.size)
        return;

    VerletConstraint* c = &constraints.data[idx];
    VerletParticle* p1 = c->particle1;
    VerletParticle* p2 = c->particle2;

    // Skip if both particles are static
    if (p1->is_static && p2->is_static)
        return;

    float2 delta = p2->position - p1->position;
    float dist = Length(delta);
    
    // Avoid division by zero
    if (dist < 0.0001f) {
        c->correction = make_float2(0.0f, 0.0f);
        return;
    }
        
    // Calculate the error as a ratio of current length to rest length
    float error = (dist - c->rest_length) / (dist);
    
    // Store the correction in the constraint
    c->correction = delta * error * c->stiffness * 0.5f;
}

// Second pass: Apply the corrections
__global__ void ApplyConstraintCorrectionsKernel(CudaWorld* cuda_world) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Array<VerletConstraint> constraints = cuda_world->constraints;
    if (idx >= constraints.size)
        return;

    VerletConstraint* c = &constraints.data[idx];
    
    // Apply the stored correction
    if (!c->particle1->is_static)
        c->particle1->position += c->correction;
        
    if (!c->particle2->is_static)
        c->particle2->position -= c->correction;
}

void UpdateVerletParticles(CudaWorld* cuda_world) 
{
    const int block_size = 256;
    const int num_blocks = (cuda_world->particles.size + block_size - 1) / block_size;
    
    // Update positions with fixed timestep
    UpdatePositionsKernel<<<num_blocks, block_size>>>(cuda_world);
    cudaDeviceSynchronize();

    // Solve constraints multiple times for stability
    const int solver_iterations = 1; 
    const int constraint_block_size = 256;
    const int num_constraint_blocks = (cuda_world->constraints.size + constraint_block_size - 1) / constraint_block_size;
    
    for (int i = 0; i < solver_iterations; i++) {
        // Split into two kernels to avoid race conditions
        CalculateConstraintCorrectionsKernel<<<num_constraint_blocks, constraint_block_size>>>(cuda_world);
        cudaDeviceSynchronize();
        
        ApplyConstraintCorrectionsKernel<<<num_constraint_blocks, constraint_block_size>>>(cuda_world);
        cudaDeviceSynchronize();
    }
}
