#pragma once

#include <cuda_runtime.h>

struct VerletParticle {
    float2 position;      // Current position
    float2 old_position;  // Previous position
    float2 acceleration;  // Current acceleration
    float mass;           // Particle mass
    float radius;         // Particle radius
    int is_static;        // 0 = movable, 1 = static (immovable)
};

// Connection between two particles (spring)
struct VerletConnection {
    int particleA;        // Index of first particle
    int particleB;        // Index of second particle
    float restLength;     // Rest length of the spring
    float stiffness;      // Spring stiffness (0-1)
};

// Boundary constraint (box, circle, etc.)
struct BoundaryConstraint {
    int type;             // 0 = box, 1 = circle
    float2 position;      // Center position (for circle) or top-left (for box)
    float2 size;          // Width/height (for box) or radius (for circle)
}; 