#pragma once

#include "AnymUtil.h"

enum GamePhase
{
    GamePhase_SimulationScreen,
};

struct SimulationSettings
{
    R32 mutation_rate = 0.012f;
    I32 energy_transfer_on_hit = 350;
    I32 charge_duration = 20;     // 60 per second.
    I32 charge_refractory_period = 240;     // 60 per second.

    // Current phase settings
    GamePhase current_phase = GamePhase_SimulationScreen;
};

inline SimulationSettings global_settings;