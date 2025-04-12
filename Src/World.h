#pragma once

#include <cuda_runtime.h>
#include "Cuda/VerletParticle.h"

#include "AnymUtil.h"
#include "Memory.h"

struct CudaWorld
{
    Array<VerletParticle> particles;
};

struct World 
{
    // GPU memory arena.
    MemoryArena* cuda_arena;
    CudaWorld* cuda_world;

    // CPU stufff
    MemoryArena* world_arena;
};

bool InitWorld(World* world);
void UpdateWorld(World* world);
void RenderWorld(World* world, Renderer* renderer);
void DestroyWorld(World* world);
