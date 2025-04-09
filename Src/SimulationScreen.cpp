#include <functional>
#include <thread>

#include <imgui.h>
#include <implot.h>
#include <gl3w.h>

#include "SimulationScreen.h"
#include "DebugInfo.h"
#include "Lucide.h"

#include "Renderer.h"

static void
DoScreenWorldRender(SimulationScreen* screen, Window* window)
{
    Shader* shader = &screen->shader;
    Mesh2D* mesh = &screen->dynamic_mesh;
    Camera2D* cam = &screen->cam;

    UseShader(&screen->shader);
    glBindTexture(GL_TEXTURE_2D, screen->atlas->handle);
    UpdateCamera(cam, window->width, window->height);
    SetTransform(shader, &cam->transform.m[0][0]);

    BufferData(mesh, GL_DYNAMIC_DRAW);
    Draw(mesh);
    Clear(mesh);
}

static void
DoDebugInfo(SimulationScreen* screen, Window* window)
{
    ImGui::Text("FPS: %.0f", window->fps);
    ImGui::Text("Update: %.2f millis", window->update_millis);
    // ImGui::Text("Camera scale: %.2f", screen->renderer->cam.scale);
    // ImGui::Text("Camera bounds pos: %.2f %.2f", screen->renderer->cam.bounds.pos.x, screen->renderer->cam.bounds.pos.y);
    // ImGui::Text("Camera bounds dim: %.2f %.2f", screen->renderer->cam.bounds.dims.x, screen->renderer->cam.bounds.dims.y);
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

    // Do verlet integration.

    // Do some test rendering
    Mesh2D* mesh = &screen->dynamic_mesh;
    UseShader(&screen->shader);
    glBindTexture(GL_TEXTURE_2D, screen->atlas->handle);
    UpdateCamera(cam, window->width, window->height);
    SetTransform(&screen->shader, &cam->transform.m[0][0]);

    PushLine(mesh, V2(0,0), V2(102,100), 4.0f, V2(0,0), V2(1,1), Color_White);

    BufferData(mesh, GL_DYNAMIC_DRAW);
    Draw(mesh);
    Clear(mesh);

    return 0;
}

void
InitSimulationScreen(SimulationScreen* screen)
{
    screen->world_arena = CreateMemoryArena(GigaBytes(1));
    screen->screen_arena = CreateMemoryArena(MegaBytes(32));

    screen->cam.pos = V2(0,0);
    screen->cam.scale = 1.0f;

    screen->update_times.FillAndSetValue(0);

    // TODO: This is the actual renderer. Keep this.
    screen->shader = CreateShader();
    screen->atlas = MakeDefaultTexture(screen->screen_arena, 511);
    screen->square = &screen->atlas->regions[1];
}
