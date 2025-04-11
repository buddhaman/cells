#include "World.h"
#include "Cuda/VerletCuda.h"
#include <cstdlib>

bool InitWorld(World* world, int num_particles) 
{
    world->num_particles = num_particles;

    cudaError_t cuda_status = cudaMallocManaged(&world->particles, num_particles * sizeof(VerletParticle));
    if (cuda_status != cudaSuccess) {
        return false;
    }
    
    // Initialize particles in a 100x100 square with random impulses
    for (int i = 0; i < num_particles; i++) 
    {
        // Random positions within 100x100 square
        float x = ((float)rand() / RAND_MAX) * 100.0f;
        float y = ((float)rand() / RAND_MAX) * 100.0f;
        
        // Random impulse direction and magnitude
        float impulse_x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float impulse_y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        
        // Set current position
        world->particles[i].position = make_float2(x, y);
        // Set old position to create initial velocity from impulse
        world->particles[i].old_position = make_float2(x - impulse_x, y - impulse_y);
        world->particles[i].radius = 0.01f;
        world->particles[i].is_static = 0;
    }

    return true;
}

void UpdateWorld(World* world) 
{
    UpdateVerletParticles(world->particles, world->num_particles);
}

void RenderWorld(World* world, Renderer* renderer)
{
    for(int i = 0; i < world->num_particles; i++)
    {
        Vec2 pos = V2(world->particles[i].position.x, world->particles[i].position.y);
        RenderCircle(renderer, pos, 3.0f, Color_White);
    }
}

void DestroyWorld(World* world) 
{
    if (world->particles) {
        cudaFree(world->particles);
        world->particles = nullptr;
    }
    world->num_particles = 0;
}
