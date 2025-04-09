#pragma once

#include "VerletParticle.h"

// Main simulation step
void VerletSimulationStep(
    VerletParticle* particles, 
    int num_particles,
    VerletConnection* connections, 
    int num_connections,
    BoundaryConstraint* boundaries, 
    int num_boundaries,
    float dt, 
    float gravity,
    int solver_iterations
);

// Apply external forces like gravity
void ApplyForces(
    VerletParticle* particles, 
    int num_particles, 
    float gravity
);

// Perform integration step
void VerletIntegration(
    VerletParticle* particles, 
    int num_particles, 
    float dt
);

// Solve constraints between particles
void SolveConstraints(
    VerletParticle* particles, 
    int num_particles,
    VerletConnection* connections, 
    int num_connections,
    BoundaryConstraint* boundaries, 
    int num_boundaries,
    int solver_iterations
); 