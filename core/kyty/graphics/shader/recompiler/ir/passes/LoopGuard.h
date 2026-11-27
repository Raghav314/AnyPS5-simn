#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

enum class LoopExit : uint8_t {
	EmptyWalk,
	Guard,
};

struct LoopGuards {
	uint32_t                                      guarded   = 0;
	uint32_t                                      bounded   = 0;
	uint32_t                                      unguarded = 0;
	std::vector<std::pair<uint32_t, const char*>> refusals;
};

const char* GuardLoop(Program& program, Block& header, uint32_t limit, LoopExit exit);
LoopGuards  GuardComputeLoops(Program& program, uint32_t limit, bool several_waves, std::span<Block* const> bounded);

}
