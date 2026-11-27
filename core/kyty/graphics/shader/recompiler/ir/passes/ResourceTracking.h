#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg,
                    ShaderStageInputInfo input, bool runtime_buffer_descriptors = false,
                    bool runtime_sampler_descriptors = false);
bool TakeRuntimeSamplerStop();

}

#endif
