#include "World.h"

#include <cstdlib>

bool InitWorld(World* world, int num_particles) 
{
    world->num_particles = num_particles;
    world->initialized = false;

    // Allocate CUDA managed memory for particles
    cudaError_t cuda_status = cudaMallocManaged(&world->particles, num_particles * sizeof(VerletParticle));
    if (cuda_status != cudaSuccess) {
        return false;
    }
    
    // Initialize particles with random positions
    for (int i = 0; i < num_particles; i++) {
        // in a cube of 100x100, randomize the position .
        world->particles[i].position = { (float)rand() / RAND_MAX * 100.0f - 50.0f, 
                                       (float)rand() / RAND_MAX * 100.0f - 50.0f };
        // Give random impulse to the particles by moving old_position.
        world->particles[i].old_position = world->particles[i].position;
        world->particles[i].old_position.x += (float)rand() / RAND_MAX * 0.1f - 0.05f;
        world->particles[i].old_position.y += (float)rand() / RAND_MAX * 0.1f - 0.05f;
        //world->particles[i].acceleration = { 0.0f, -9.8f }; // Gravity
    }

    world->initialized = true;
    return true;
}

void UpdateWorld(World* world)
{
    if (!world->initialized) return;

    R32 dt = 1.0f/60.0f;

    VerletIntegration(world->particles, world->num_particles, dt);
    cudaDeviceSynchronize(); // Ensure GPU computations are complete
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
    world->initialized = false;
}
