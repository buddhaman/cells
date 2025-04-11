#include <imgui.h>
#include <implot.h>

#include "DebugInfo.h"

void
ImGuiMemoryArena(MemoryArena* arena, const char* name)
{
    R32 used = ((R32)arena->used / (1024.0f*1024.0f));
    R32 total = ((R32)arena->size / (1024.0f*1024.0f));
    ImGui::Text("%s : (%.2f / %.2f) mb used", name, used, total);
}
