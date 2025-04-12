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
    
    // Initialize particles in a true triangular lattice
    int grid_width = 80;  // Number of particles in a row
    int grid_height = 80; // Number of particles in a column
    float spacing = 10.0f; // Distance between neighboring particles
    
    // Calculate row height for equilateral triangles
    float row_height = spacing * 0.866f; // sqrt(3)/2 * spacing
    
    // Create particles in a triangular grid
    for (int y = 0; y < grid_height; y++) 
    {
        // Offset every other row by half the spacing
        float row_offset = (y % 2) * spacing * 0.5f;
        
        for (int x = 0; x < grid_width; x++) 
        {
            VerletParticle* particle = cuda_world->particles.PushBack();
            
            // Calculate position with offset for odd rows
            float px = x * spacing + row_offset;
            float py = y * row_height;
            
            particle->position = make_float2(px, py);
            particle->old_position = make_float2(px, py);
            particle->radius = 1.5f;
            particle->mass = 1.0f;
            particle->is_static = 0;
        }
    }

    // Connect particles with constraints to form triangles
    for (int y = 0; y < grid_height; y++) 
    {
        bool even_row = (y % 2) == 0;
        
        for (int x = 0; x < grid_width; x++) 
        {
            int current_idx = y * grid_width + x;
            VerletParticle* current = &cuda_world->particles.data[current_idx];
            
            // Connect horizontally to right neighbor
            if (x < grid_width - 1) 
            {
                VerletConstraint* constraint = cuda_world->constraints.PushBack();
                constraint->particle1 = current;
                constraint->particle2 = &cuda_world->particles.data[current_idx + 1];
                constraint->rest_length = spacing;
                constraint->stiffness = 0.9f;
            }
            
            // Connect to particles in the row below
            if (y < grid_height - 1) 
            {
                int below_row_idx = (y + 1) * grid_width;
                
                if (even_row) 
                {
                    // Even row - connect to below and below-right
                    // Connect downward
                    if (x < grid_width) 
                    {
                        VerletConstraint* constraint = cuda_world->constraints.PushBack();
                        constraint->particle1 = current;
                        constraint->particle2 = &cuda_world->particles.data[below_row_idx + x];
                        constraint->rest_length = row_height;
                        constraint->stiffness = 0.9f;
                    }
                    
                    // Connect down-right (diagonal)
                    if (x < grid_width - 1) 
                    {
                        VerletConstraint* constraint = cuda_world->constraints.PushBack();
                        constraint->particle1 = current;
                        constraint->particle2 = &cuda_world->particles.data[below_row_idx + x + 1];
                        constraint->rest_length = spacing;
                        constraint->stiffness = 0.9f;
                    }
                } 
                else 
                {
                    // Odd row - connect to below-left and below
                    // Connect down-left (diagonal)
                    if (x > 0) 
                    {
                        VerletConstraint* constraint = cuda_world->constraints.PushBack();
                        constraint->particle1 = current;
                        constraint->particle2 = &cuda_world->particles.data[below_row_idx + x - 1];
                        constraint->rest_length = spacing;
                        constraint->stiffness = 0.9f;
                    }
                    
                    // Connect downward
                    if (x < grid_width) 
                    {
                        VerletConstraint* constraint = cuda_world->constraints.PushBack();
                        constraint->particle1 = current;
                        constraint->particle2 = &cuda_world->particles.data[below_row_idx + x];
                        constraint->rest_length = row_height;
                        constraint->stiffness = 0.9f;
                    }
                }
            }
        }
    }

    // Add an impulse to create a wave effect
    int center_x = grid_width / 2;
    int center_y = grid_height / 2;
    int center_idx = center_y * grid_width + center_x;
    cuda_world->particles.data[center_idx].position.y += 30.0f;

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

