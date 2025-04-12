#pragma once

#include "../Array.h"

struct VerletParticle 
{
    float2 position;      
    float2 old_position; 
    float radius;       
    float mass;        
    int is_static;       
    float2 correction_accumulator;
};

struct VerletConstraint 
{
    VerletParticle* particle1;
    VerletParticle* particle2;
    float rest_length;
    float stiffness;
};

struct CudaWorld
{
    Array<VerletParticle> particles;
    Array<VerletConstraint> constraints;
};

void UpdateVerletParticles(CudaWorld* cuda_world);
