#pragma once

#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t OracleLoop    = 16u;
constexpr uint32_t OracleResults = 32u;

[[nodiscard]] constexpr uint64_t EntryFlagOf(uint64_t prologue, bool active) noexcept {
	return active ? prologue : 0u;
}
[[nodiscard]] constexpr uint64_t EntryMaskOf(uint64_t prologue, uint64_t active) noexcept {
	return prologue & active;
}
struct EntrySample {
	const char*                        failure = nullptr;
	std::vector<uint8_t>               active;
	std::vector<std::vector<uint64_t>> flags;
};
[[nodiscard]] EntrySample SampleWalkEntry(Program& program, const NativeTraversalOptions& options, Block* header, Block* latch,
                                          std::span<Inst* const> served, uint64_t seed);
[[nodiscard]] bool RewriteResultsWalk(Program& program, const NativeTraversalOptions& options, Block* header, Block* latch,
                                      std::span<Inst* const> served, NativeTraversals& result, uint32_t site);

}
