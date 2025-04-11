#pragma once

#include "Mesh2D.h"
#include "Shader.h"
#include "Window.h"
#include "SimulationSettings.h"
#include "ThreadPool.h"
#include "Texture.h"

struct SimulationScreen
{
    Mesh2D dynamic_mesh;

    R32 extra_selection_radius = 1.0f;
    MemoryArena* screen_arena;
    MemoryArena* world_arena;

    Camera2D cam;

    Renderer* renderer;

    DynamicArray<R32> update_times;

    bool isPaused = false;
    int updates_per_frame = 1;
    R32 time = 0.0f;

    SimulationScreen() : update_times(120) {} 
};

int
UpdateSimulationScreen(SimulationScreen* screen, Window* window);

void
InitSimulationScreen(SimulationScreen* screeen);