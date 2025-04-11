#pragma once

#include <cuda_runtime.h>
#include "Cuda/VerletParticle.h"

#include "AnymUtil.h"

struct World 
{
    VerletParticle* particles;
    int num_particles;
    bool initialized;
};

bool InitWorld(World* world, int num_particles);
void UpdateWorld(World* world);
void RenderWorld(World* world, Renderer* renderer);
void DestroyWorld(World* world);
