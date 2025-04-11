#pragma once

#include "VerletParticle.h"

// Main simulation step
void VerletSimulationStep(
    VerletParticle* particles, 
    int num_particles,
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

void UpdateVerletParticles(VerletParticle* particles, int num_particles);
void VerletSimulationStep(VerletParticle* particles, int num_particles);
                         