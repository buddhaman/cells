#include "VerletCuda.h"
#include <device_launch_parameters.h>

// Helper functions for vector operations
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

__device__ float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ float2 normalize(float2 v) {
    float len = length(v);
    if (len > 0.0001f) {
        return make_float2(v.x / len, v.y / len);
    }
    return make_float2(0.0f, 0.0f);
}

// Kernel to apply forces (e.g., gravity)
__global__ void ApplyForcesKernel(VerletParticle* particles, int num_particles, float gravity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    VerletParticle& p = particles[idx];
    
    // Reset acceleration
    p.acceleration = make_float2(0.0f, 0.0f);
    
    // Apply gravity if particle is not static
    if (!p.is_static) {
        p.acceleration.y += gravity;
    }
}

// CUDA kernel for Verlet integration
__global__ void VerletIntegrationKernel(VerletParticle* particles, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || particles[idx].is_static) return;

    VerletParticle& p = particles[idx];
    
    // Store current position
    float2 temp = p.position;
    
    // Verlet integration
    p.position = p.position + (p.position - p.old_position) + p.acceleration * (dt * dt);
    
    // Update old position
    p.old_position = temp;
}

// Solve particle connections (springs)
__global__ void SolveConnectionsKernel(VerletParticle* particles, int num_particles, 
                                      VerletConnection* connections, int num_connections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_connections) return;

    VerletConnection& connection = connections[idx];
    
    if (connection.particleA >= num_particles || connection.particleB >= num_particles) return;
    
    VerletParticle& p1 = particles[connection.particleA];
    VerletParticle& p2 = particles[connection.particleB];
    
    // Calculate distance vector
    float2 delta = p2.position - p1.position;
    float distance = length(delta);
    
    // Skip if particles are at the same position
    if (distance < 0.0001f) return;
    
    // Calculate correction based on stiffness and rest length
    float difference = (distance - connection.restLength) / distance;
    float2 correction = delta * (difference * 0.5f * connection.stiffness);
    
    // Apply correction if particles are not static
    if (!p1.is_static) p1.position = p1.position + correction;
    if (!p2.is_static) p2.position = p2.position - correction;
}

// Solve collisions between particles
__global__ void SolveParticleCollisionsKernel(VerletParticle* particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || particles[idx].is_static) return;
    
    VerletParticle& p1 = particles[idx];
    
    // Check collisions with all other particles (n^2 complexity)
    for (int i = 0; i < num_particles; i++) {
        if (i == idx) continue; // Skip self
        
        VerletParticle& p2 = particles[i];
        
        // Calculate distance vector between particles
        float2 delta = p2.position - p1.position;
        float distance = length(delta);
        
        // Minimum distance to maintain (sum of radii)
        float minDist = p1.radius + p2.radius;
        
        // If too close, push particles apart
        if (distance < minDist && distance > 0.0001f) {
            float2 correction = delta * ((minDist - distance) / distance * 0.5f);
            
            if (!p1.is_static) p1.position = p1.position - correction;
            if (!p2.is_static) p2.position = p2.position + correction;
        }
    }
}

// Solve boundary constraints
__global__ void SolveBoundaryConstraintsKernel(VerletParticle* particles, int num_particles,
                                             BoundaryConstraint* boundaries, int num_boundaries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || particles[idx].is_static) return;
    
    VerletParticle& p = particles[idx];
    
    // Check against all boundaries
    for (int i = 0; i < num_boundaries; i++) {
        BoundaryConstraint& boundary = boundaries[i];
        
        if (boundary.type == 0) { // Box constraint
            // Left wall
            if (p.position.x - p.radius < boundary.position.x) {
                p.position.x = boundary.position.x + p.radius;
            }
            
            // Right wall
            if (p.position.x + p.radius > boundary.position.x + boundary.size.x) {
                p.position.x = boundary.position.x + boundary.size.x - p.radius;
            }
            
            // Top wall
            if (p.position.y - p.radius < boundary.position.y) {
                p.position.y = boundary.position.y + p.radius;
            }
            
            // Bottom wall
            if (p.position.y + p.radius > boundary.position.y + boundary.size.y) {
                p.position.y = boundary.position.y + boundary.size.y - p.radius;
            }
        } 
        else if (boundary.type == 1) { // Circle constraint
            float2 center = boundary.position;
            float radius = boundary.size.x; // Use x as radius
            
            float2 delta = p.position - center;
            float distance = length(delta);
            
            if (distance > radius - p.radius) {
                // Constraint to be inside the circle
                p.position = center + normalize(delta) * (radius - p.radius);
            }
        }
    }
}

// Host-side functions
void ApplyForces(VerletParticle* particles, int num_particles, float gravity) {
    const int blockSize = 256;
    const int numBlocks = (num_particles + blockSize - 1) / blockSize;
    
    ApplyForcesKernel<<<numBlocks, blockSize>>>(particles, num_particles, gravity);
    cudaDeviceSynchronize();
}

void VerletIntegration(VerletParticle* particles, int num_particles, float dt) {
    const int blockSize = 256;
    const int numBlocks = (num_particles + blockSize - 1) / blockSize;
    
    VerletIntegrationKernel<<<numBlocks, blockSize>>>(particles, num_particles, dt);
    cudaDeviceSynchronize();
}

void SolveConstraints(VerletParticle* particles, int num_particles,
                     VerletConnection* connections, int num_connections,
                     BoundaryConstraint* boundaries, int num_boundaries,
                     int solver_iterations) {
    
    const int particleBlockSize = 256;
    const int particleNumBlocks = (num_particles + particleBlockSize - 1) / particleBlockSize;
    
    const int connectionBlockSize = 256;
    const int connectionNumBlocks = (num_connections + connectionBlockSize - 1) / connectionBlockSize;
    
    // Solve constraints multiple times to improve accuracy
    for (int i = 0; i < solver_iterations; i++) {
        // Solve connections (springs)
        if (num_connections > 0) {
            SolveConnectionsKernel<<<connectionNumBlocks, connectionBlockSize>>>(
                particles, num_particles, connections, num_connections);
            cudaDeviceSynchronize();
        }
        
        // Solve particle collisions
        SolveParticleCollisionsKernel<<<particleNumBlocks, particleBlockSize>>>(
            particles, num_particles);
        cudaDeviceSynchronize();
        
        // Solve boundary constraints
        if (num_boundaries > 0) {
            SolveBoundaryConstraintsKernel<<<particleNumBlocks, particleBlockSize>>>(
                particles, num_particles, boundaries, num_boundaries);
            cudaDeviceSynchronize();
        }
    }
}

// Main simulation step
void VerletSimulationStep(VerletParticle* particles, int num_particles,
                         VerletConnection* connections, int num_connections,
                         BoundaryConstraint* boundaries, int num_boundaries,
                         float dt, float gravity, int solver_iterations) {
    
    // Step 1: Apply external forces
    ApplyForces(particles, num_particles, gravity);
    
    // Step 2: Integrate positions
    VerletIntegration(particles, num_particles, dt);
    
    // Step 3: Solve constraints
    SolveConstraints(particles, num_particles, connections, num_connections,
                   boundaries, num_boundaries, solver_iterations);
} 