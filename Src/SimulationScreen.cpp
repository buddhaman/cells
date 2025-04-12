#include <functional>
#include <thread>

#include <imgui.h>
#include <implot.h>
#include <gl3w.h>

#include "SimulationScreen.h"
#include "DebugInfo.h"
#include "Lucide.h"
#include "Profiling.h"

#include "Renderer.h"

static void
DoProfilerWindow(SimulationScreen* screen)
{
    if (ImGui::Button("Add impulse"))
    {
        screen->world->cuda_world->particles.data[0].position.x += 20.0f;
        screen->world->cuda_world->particles.data[0].position.y += 20.0f;

        screen->world->cuda_world->particles.data[10000].position.x -= 20.0f;
        screen->world->cuda_world->particles.data[10000].position.y -= 20.0f;
    }
    
    ImGui::Columns(4, "profiler_columns");
    ImGui::SetColumnWidth(0, 200);
    ImGui::Text("Section"); ImGui::NextColumn();
    ImGui::Text("Avg (ms)"); ImGui::NextColumn();
    ImGui::Text("Min (ms)"); ImGui::NextColumn();
    ImGui::Text("Max (ms)"); ImGui::NextColumn();
    ImGui::Separator();
    
    // Just iterate directly over the map elements
    for (auto it = ProfileScope::map.begin(); it != ProfileScope::map.end(); ++it)
    {
        ProfilerMetric* metric = it->second;
        ImGui::Text("%s", metric->name); ImGui::NextColumn();
        ImGui::Text("%.2f", (float)metric->samples.GetAverage()); ImGui::NextColumn();
        ImGui::Text("%.2f", (float)metric->samples.GetMin()); ImGui::NextColumn();
        ImGui::Text("%.2f", (float)metric->samples.GetMax()); ImGui::NextColumn();
    }
    
    ImGui::Columns(1);
}

void
DoStatisticsWindow(SimulationScreen* screen)
{
    ImPlotFlags updatetime_plot_flags = ImPlotFlags_NoBoxSelect | 
                        ImPlotFlags_NoInputs | 
                        ImPlotFlags_NoFrame | 
                        ImPlotFlags_NoLegend;

    ImPlot::SetNextAxesLimits(0, (int)screen->update_times.size, 0, 80.0f, 0);
    if(ImPlot::BeginPlot("Frame update time", V2(-1, 200), updatetime_plot_flags))
    { 
        ImPlot::PlotBars("Update time", screen->update_times.data, (int)screen->update_times.size, 1);
        ImPlot::EndPlot();
    }

}

static void
DoDebugInfo(SimulationScreen* screen, Window* window)
{
    ImGui::Text("FPS: %.0f", window->fps);
    ImGui::Text("Update: %.2f millis", window->update_millis);
    DoStatisticsWindow(screen);
    DoProfilerWindow(screen);
}

int
UpdateSimulationScreen(SimulationScreen* screen, Window* window)
{
    InputHandler* input = &window->input;
    Camera2D* cam = &screen->cam;
    UpdateCamera(cam, window->width, window->height);

    screen->update_times.Shift(-1);
    screen->update_times[screen->update_times.size-1] = window->update_millis;

    // TODO: Do not hardcode this. 1.0f/60.0f;
    screen->time +=  1.0f/60.0f;

    ImGui::Begin(ICON_LC_BUG "Debug info");
    DoDebugInfo(screen, window);
    ImGui::End();

    {
        PROFILE_SCOPE("world update");
        // Do verlet integration.
        UpdateWorld(screen->world);
    }

    Renderer* renderer = screen->renderer;

    {
        PROFILE_SCOPE(" world render");
        RenderWorld(screen->world, renderer);
    }

    {
        PROFILE_SCOPE("renderer update");
        Render(renderer, cam, window->width, window->height);
    }

    return 0;
}

void
InitSimulationScreen(SimulationScreen* screen)
{
    //screen->world_arena = CreateMemoryArena(GigaBytes(1));
    screen->screen_arena = CreateMemoryArena(MegaBytes(32));

    screen->cam.pos = V2(400,400);
    screen->cam.scale = 1.0f;

    screen->world = PushNewStruct(screen->screen_arena, World);
    InitWorld(screen->world);

    screen->update_times.FillAndSetValue(0);

    screen->renderer = CreateRenderer(screen->screen_arena);
}
