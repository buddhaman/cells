#include "External.h"
#include "External.cpp"
#include <cuda_runtime.h>
#include "Cuda/VerletCuda.h"

#include "AnymUtil.h"
#include "Array.h"
#include "TiltCamera.h"
#include "Camera2D.h"
#include "DebugInfo.h"
#include "External.h"
#include "InputHandler.h"
#include "Linalg.h"
#include "Noise.h"
#include "TMath.h"
#include "TString.h"
#include "Memory.h"
#include "Mesh2D.h"
#include "Shader.h"
#include "Texture.h"
#include "Renderer.h"
#include "Brain.h"
#include "SimulationScreen.h"
#include "ThreadPool.h"

#include "Brain.cpp"
#include "Array.h"
#include "Camera2D.cpp"
#include "TiltCamera.cpp"
#include "DebugInfo.cpp"
#include "InputHandler.cpp"
#include "Linalg.cpp"
#include "Memory.cpp"
#include "Mesh2D.cpp"
#include "Shader.cpp"
#include "Texture.cpp"
#include "Renderer.cpp"
#include "SimulationScreen.cpp"
#include "SimulationSettings.h"
#include "TimVectorMath.cpp"
#include "Window.cpp"

#include <direct.h>  

static void
PrintWorkingDirectory()
{
    char cwd[1024];  

    if (_getcwd(cwd, sizeof(cwd))) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        std::cerr << "Error retrieving the current working directory." << std::endl;
    }
}

int main(int argc, char** argv) 
{
    // Initialize random seed once. This will be replaced by my own rng.
    srand((U32)(time(nullptr)));

    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    // Test Verlet integration
    const int num_particles = 1000;
    VerletParticle* particles;
    cudaMallocManaged(&particles, num_particles * sizeof(VerletParticle));
    
    // Initialize particles
    for (int i = 0; i < num_particles; i++) {
        particles[i].position = { (float)rand() / RAND_MAX * 2.0f - 1.0f, 
                                (float)rand() / RAND_MAX * 2.0f - 1.0f };
        particles[i].old_position = particles[i].position;
        particles[i].acceleration = { 0.0f, -9.8f }; // Gravity
    }

#undef CreateWindow
    Window* window = CreateWindow(1280, 720);
    if(!window) 
    {
        cudaFree(particles);
        return -1;
    }
    window->fps = 60;

    PrintWorkingDirectory();

    SimulationScreen screen;
    InitSimulationScreen(&screen);

    float dt = 1.0f / 60.0f; // Time step

    while (window->running) 
    {
        WindowBegin(window);
        
        VerletIntegration(particles, num_particles, dt);
        
        switch(global_settings.current_phase)
        {
            case GamePhase_SimulationScreen: 
            {
                UpdateSimulationScreen(&screen, window);
            } break;
        }
        WindowEnd(window);
    }

    // Cleanup
    cudaFree(particles);
    DestroyWindow(window);

    return 0;
}
