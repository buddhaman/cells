#pragma once

#include <cuda_runtime.h>

#include "AnymUtil.h"
#include "Memory.h"
#include "Array.h"
#include "Renderer.h"

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
