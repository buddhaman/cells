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

    int max_particles = 100000;
    CudaWorld* cuda_world = world->cuda_world = PushNewStruct(cuda_arena, CudaWorld);

    cuda_world->particles = CreateArray<VerletParticle>(cuda_arena, max_particles);
    cuda_world->constraints = CreateArray<VerletConstraint>(cuda_arena, max_particles*2);
    
    // Initialize particles in a triangular (hexagonal) grid
    int grid_size = 100; // Number of particles in a row
    float spacing = 8.0f; // Distance between neighboring particles
    
    // For a triangular lattice:
    // - Every other row is offset by half spacing
    // - Vertical distance between rows is sqrt(3)/2 * spacing
    float row_height = sqrtf(3.0f) * 0.5f * spacing;
    
    // Create particles in a triangular grid
    for (int y = 0; y < grid_size; y++) 
    {
        float row_offset = (y % 2) * 0.5f * spacing; // Offset every other row
        for (int x = 0; x < grid_size; x++) 
        {
            VerletParticle* particle = cuda_world->particles.PushBack();
            float px = x * spacing + row_offset;
            float py = y * row_height;
            particle->position = make_float2(px, py);
            particle->old_position = make_float2(px, py);
            particle->radius = 1.0f;
            particle->mass = 1.0f;
            particle->is_static = 0;
        }
    }

    // Connect particles with constraints to form triangles
    for (int y = 0; y < grid_size; y++) 
    {
        bool odd_row = y % 2 == 1;
        for (int x = 0; x < grid_size; x++) 
        {
            int current_idx = y * grid_size + x;
            VerletParticle* current = &cuda_world->particles.data[current_idx];
            
            // Connect to right neighbor (horizontal constraints)
            if (x < grid_size - 1) 
            {
                VerletConstraint* constraint = cuda_world->constraints.PushBack();
                constraint->particle1 = current;
                constraint->particle2 = &cuda_world->particles.data[current_idx + 1];
                constraint->rest_length = spacing;
                constraint->stiffness = 0.8f;
            }
            
            // Connect to lower neighbors (forms triangles)
            if (y < grid_size - 1) 
            {
                // In a triangular grid, each point connects to either 2 or 1 particles
                // in the row below, depending on if it's an odd or even row
                
                // Connect to particle directly below (diagonally)
                int below_idx = (y + 1) * grid_size + x;
                if (x < grid_size - odd_row) // Adjust for offset rows
                {
                    VerletConstraint* constraint = cuda_world->constraints.PushBack();
                    constraint->particle1 = current;
                    constraint->particle2 = &cuda_world->particles.data[below_idx];
                    constraint->rest_length = row_height;
                    constraint->stiffness = 0.8f;
                }
                
                // Connect to bottom-right neighbor
                if ((odd_row && x < grid_size - 1) || (!odd_row && x < grid_size - 1))
                {
                    VerletConstraint* constraint = cuda_world->constraints.PushBack();
                    constraint->particle1 = current;
                    constraint->particle2 = &cuda_world->particles.data[below_idx + 1];
                    constraint->rest_length = sqrtf(row_height*row_height + (0.5f*spacing)*(0.5f*spacing));
                    constraint->stiffness = 0.8f;
                }
            }
        }
    }

    // add random impulse to first particle
    // cuda_world->particles.data[0].position.x += 40.0f;
    // cuda_world->particles.data[0].position.y += 1.1f;

    return true;
}

void UpdateWorld(World* world) 
{
    CudaWorld* cuda_world = world->cuda_world;
    UpdateVerletParticles(cuda_world);
}

void RenderWorld(World* world, Renderer* renderer)
{
    // Render constraints
    for(VerletConstraint& constraint : world->cuda_world->constraints)
    {
        Vec2 pos1 = V2(constraint.particle1->position.x, constraint.particle1->position.y);
        Vec2 pos2 = V2(constraint.particle2->position.x, constraint.particle2->position.y);
        
        R32 current_length = V2Len(pos2 - pos1);
        R32 stretch_diff = constraint.rest_length - current_length;
        
        // Color based on stretch difference using HSV color space
        R32 t = Clamp(-1.0f, stretch_diff*16.0f, 1.0f);
        R32 hue = Lerp((t + 1.0f) * 0.5f, 240.0f, 0.0f);
        R32 saturation = 1.0f;
        R32 value = 1.0f;
        U32 color = HSVAToRGBA(hue, saturation, value, 1.0f);
        RenderLine(renderer, pos1, pos2, 1.0f, color);
    }

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

