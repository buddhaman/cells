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
#include "World.cpp"
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

#undef CreateWindow
    Window* window = CreateWindow(1800, 900);
    if(!window) 
    {
        return -1;
    }
    window->fps = 60;

    PrintWorkingDirectory();

    SimulationScreen screen;
    InitSimulationScreen(&screen);

    while (window->running) 
    {
        WindowBegin(window);
        
        switch(global_settings.current_phase)
        {
            case GamePhase_SimulationScreen: 
            {
                UpdateSimulationScreen(&screen, window);
            } break;
        }
        WindowEnd(window);
    }

    DestroyWindow(window);
    DestroyWorld(screen.world);

    return 0;
}
