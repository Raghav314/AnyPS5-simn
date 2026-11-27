#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t OracleRemaining = 1u;
constexpr uint32_t OracleForms     = 2u;
constexpr uint32_t OracleOnly      = 4u;
constexpr uint32_t OraclePixel     = 8u;

[[nodiscard]] bool OracleLaneWalk(std::span<Inst* const> walk);
uint32_t           MarkOracleWalk(std::span<Inst* const> walk);
size_t             MarkEmptyWalk(const Program& program, std::span<Inst* const> intersections,
                                 const NativeTraversalOptions& options);
struct EmptyWalkBounds {
	size_t              empty   = 0;
	size_t              native  = 0;
	size_t              refused = 0;
	std::vector<Block*> headers;
};

size_t             MarkWalksEmpty(Program& program);
EmptyWalkBounds    BoundEmptyWalks(Program& program, uint32_t limit, uint32_t native_limit);

}
