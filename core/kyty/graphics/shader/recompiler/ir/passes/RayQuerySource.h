#pragma once

#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct OracleWalks {
	std::vector<NativeTraversals>          sources;
	std::vector<std::pair<Block*, Block*>> loops;
	std::vector<RayQueryExtraSource>       headers;
	std::vector<std::vector<uint32_t>>     loop_headers;
	std::vector<OracleImprovement>         improvements;
	const char*                            refusal = nullptr;
	uint32_t                               detail  = 0;
};

[[nodiscard]] OracleWalks DeriveOracleWalks(const Program& program, std::span<Inst* const> intersections);

}
