#include "World.h"
#include "Cuda/VerletCuda.h"
#include <cstdlib>

bool InitWorld(World* world)
{
    // Make empty arena and fill with gpu memory.
    MemoryArena* cuda_arena = world->cuda_arena = new MemoryArena();
    cuda_arena->size = MegaBytes(512);
    cudaError_t cuda_status = cudaMallocManaged(&cuda_arena->base, cuda_arena->size);
    if (cuda_status != cudaSuccess) 
    {
        printf("Failed to allocate cuda memory: %s\n", cudaGetErrorString(cuda_status));
        return false;
    }

    int max_particles = 1000;
    CudaWorld* cuda_world = world->cuda_world = PushNewStruct(cuda_arena, CudaWorld);
    cuda_world->particles = CreateArray<VerletParticle>(cuda_arena, max_particles);
    
    // Initialize particles in a 100x100 square with random impulses
    for (int i = 0; i < max_particles; i++) 
    {
        // Random positions within 100x100 square
        float x = ((float)rand() / RAND_MAX) * 100.0f;
        float y = ((float)rand() / RAND_MAX) * 100.0f;
        
        // Random impulse direction and magnitude
        float impulse_x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float impulse_y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        
        // Set current position
        VerletParticle* particle = cuda_world->particles.PushBack();
        particle->position = make_float2(x, y);
        // Set old position to create initial velocity from impulse
        particle->old_position = make_float2(x - impulse_x, y - impulse_y);
        particle->radius = 1.0f;
        particle->mass = 1.0f;
        particle->is_static = 0;
    }

    return true;
}

void UpdateWorld(World* world) 
{
    CudaWorld* cuda_world = world->cuda_world;
    UpdateVerletParticles(cuda_world->particles.data, (int)cuda_world->particles.size);
}

void RenderWorld(World* world, Renderer* renderer)
{
    for(VerletParticle& particle : world->cuda_world->particles)
    {
        Vec2 pos = V2(particle.position.x, particle.position.y);
        RenderCircle(renderer, pos, particle.radius, Color_White);
    }
}

void DestroyWorld(World* world) 
{
    cudaFree(world->cuda_arena->base);
}

