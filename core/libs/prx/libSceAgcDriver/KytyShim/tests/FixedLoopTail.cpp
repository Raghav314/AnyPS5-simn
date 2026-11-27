#include "graphics/shader/recompiler/ir/passes/LoopGuard.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <map>
#include <memory>

namespace {

using namespace Libs::Graphics::ShaderRecompiler;
using namespace IR;
using Op = ValueOpcode;
using Wave = std::array<uint32_t, 64>;

enum class Variant { Plain, Coherent, Atomic, Barrier, TwoWaves, OuterLane, ScalarLane, OtherMask, EscapeStep, NoBallot };

struct Inputs {
	Wave start;
	Wave active;
};

struct Result {
	Wave cursor;
	Wave payload;
	uint32_t marks;
	uint32_t turns;
	bool rewritten;
	bool guarded;
};

Result Run(const Inputs& input, Variant variant) {
	constexpr uint32_t limit = 16;
	Program program;
	for (uint32_t id = 0; id < 5; ++id) {
		auto& block = *program.block_storage.emplace_back(std::make_unique<Block>());
		block.id = id;
		program.blocks.push_back(&block);
	}
	auto& entry = *program.blocks[0];
	auto& header = *program.blocks[1];
	auto& test = *program.blocks[2];
	auto& latch = *program.blocks[3];
	auto& exit = *program.blocks[4];
	auto link = [](Block& from, Block& to) {
		from.terminator.kind = CFG::TerminatorKind::Branch;
		from.terminator.true_block = &to;
		from.AddBranch(&to);
	};
	link(entry, header);
	link(header, test);
	link(test, latch);
	link(latch, header);
	header.terminator.loop_header = true;
	header.terminator.merge_block = &exit;
	header.terminator.continue_block = &latch;
	test.terminator.kind = CFG::TerminatorKind::ConditionalBranch;
	test.terminator.false_block = &exit;
	test.AddBranch(&exit);

	const auto input_at = [&](uint32_t index) {
		return Value(&entry.AppendNewInst(Op::GetUserData, {Value(static_cast<ScalarReg>(index))}));
	};
	const auto start = input_at(0);
	const auto mask = input_at(1);
	const Value active(&entry.AppendNewInst(Op::INotEqual32, {mask, Value(0u)}));
	auto& cursor = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
	auto& payload = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
	cursor.AddPhiOperand(&entry, start);
	payload.AddPhiOperand(&entry, Value(0u));
	const Value unfinished(&test.AppendNewInst(Op::INotEqual32, {Value(&payload), Value(77u)}));
	const Value live(&test.AppendNewInst(Op::LogicalAnd, {active, unfinished}));
	test.condition = variant == Variant::NoBallot ? Value(true) : Value(&test.AppendNewInst(Op::ConditionRef, {live},
	                       static_cast<uint64_t>(CFG::BranchCondition::ExecNonZero)));
	if (variant == Variant::OuterLane || variant == Variant::ScalarLane) {
		const Value lane(&entry.AppendNewInst(Op::GetBuiltin,
		                                     {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationIndex)), Value(0u)}));
		const Value low(&entry.AppendNewInst(Op::ULessThan32, {lane, Value(32u)}));
		auto& join = *program.block_storage.emplace_back(std::make_unique<Block>());
		join.id = program.blocks.size();
		program.blocks.push_back(&join);
		link(exit, join);
		entry.terminator.kind = CFG::TerminatorKind::ConditionalBranch;
		entry.terminator.false_block = &join;
		entry.terminator.merge_block = &join;
		entry.condition = variant == Variant::OuterLane ? low :
		    Value(&entry.AppendNewInst(Op::ConditionRef, {low}, static_cast<uint64_t>(CFG::BranchCondition::ScalarInstruction)));
		entry.AddBranch(&join);
	}
	auto& memory = program.memory_info.emplace_back();
	memory.kind = ResourceKind::IndirectBuffer;
	memory.idxen = true;
	memory.coherent = variant == Variant::Coherent;
	const Value resource(&latch.AppendNewInst(Op::GetBufferResource,
	                                         {Value(0u), Value(0u), Value(1u), Value(1u << 12u)}));
	const auto read_mask = variant == Variant::OtherMask ? active : live;
	const Value loaded(&latch.AppendNewInst(Op::LoadBufferU32,
	                                       {resource, Value(&cursor), Value(0u), Value(0u), read_mask}));
	payload.AddPhiOperand(&latch, Value(&latch.AppendNewInst(Op::SelectU32, {live, loaded, Value(&payload)})));
	const Value increment(&latch.AppendNewInst(Op::IAdd32, {Value(&cursor), Value(1u)}));
	const Value step(&latch.AppendNewInst(Op::SelectU32, {live, increment, Value(&cursor)}));
	cursor.AddPhiOperand(&latch, step);
	exit.AppendNewInst(Op::ReferenceU32, {Value(&cursor)});
	if (variant == Variant::EscapeStep) {
		auto& carried = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
		carried.AddPhiOperand(&entry, Value(0u));
		carried.AddPhiOperand(&latch, step);
		exit.AppendNewInst(Op::ReferenceU32, {Value(&carried)});
	}
	if (variant == Variant::Barrier) latch.AppendNewInst(Op::Barrier);
	if (variant == Variant::Atomic) {
		latch.AppendNewInst(Op::BufferAtomicIAdd32, {resource, Value(&cursor), Value(0u), Value(1u), Value(0u), live});
	}

	const auto guards = GuardComputeLoops(program, limit, variant == Variant::TwoWaves, {});
	Result result {};
	result.rewritten = cursor.Arg(1) != step;
	result.guarded = guards.guarded == 1;
	if (!result.guarded || variant == Variant::OuterLane || variant == Variant::ScalarLane ||
	    variant == Variant::OtherMask || variant == Variant::EscapeStep || variant == Variant::NoBallot) return result;

	std::map<const Inst*, Wave> values;
	values[start.Instruction()] = input.start;
	values[mask.Instruction()] = input.active;
	values[active.Instruction()] = input.active;
	const auto read = [&](Value value, size_t lane) -> uint32_t {
		if (const auto* inst = value.TryInstruction()) return values.at(inst)[lane];
		return value.GetType() == Type::U1 ? value.U1() : value.U32();
	};
	const auto run = [&](const Block& block, const Block* from) {
		std::map<const Inst*, Wave> phis;
		for (auto& inst: block) {
			if (inst.GetOpcode() != Op::Phi) break;
			const auto val = inst.Arg(inst.PhiBlock(0) == from ? 0 : 1);
			for (size_t i = 0; i < result.cursor.size(); ++i) phis[&inst][i] = read(val, i);
		}
		for (const auto& [inst, wave]: phis) values[inst] = wave;
		for (auto& inst: block) {
			const auto op = inst.GetOpcode();
			if (op == Op::Phi || op == Op::GetBufferResource || op == Op::Barrier || op == Op::BufferAtomicIAdd32) continue;
			auto& output = values[&inst];
			if (op == Op::ConditionRef) {
				const auto& predicate = values.at(inst.Arg(0).Instruction());
				switch (inst.Flags<CFG::BranchCondition>()) {
					case CFG::BranchCondition::ExecNonZero:
						output.fill(std::ranges::any_of(predicate, [](uint32_t v) { return v != 0; }));
						break;
					case CFG::BranchCondition::ScalarInstruction: output = predicate; break;
					default: throw std::logic_error("unsupported vote in fixed loop test");
				}
				continue;
			}
			for (size_t i = 0; i < output.size(); ++i) {
				const auto arg = [&](size_t n) { return read(inst.Arg(n), i); };
				switch (op) {
					case Op::LoadBufferU32: output[i] = arg(4) && arg(1) == 0 ? 77u : 0u; break;
					case Op::SelectU32: output[i] = arg(0) ? arg(1) : arg(2); break;
					case Op::IAdd32: output[i] = arg(0) + arg(1); break;
					case Op::ISub32: output[i] = arg(0) - arg(1); break;
					case Op::IEqual32: output[i] = arg(0) == arg(1); break;
					case Op::INotEqual32: output[i] = arg(0) != arg(1); break;
					case Op::ULessThan32: output[i] = arg(0) < arg(1); break;
					case Op::ULessThanEqual32: output[i] = arg(0) <= arg(1); break;
					case Op::UGreaterThanEqual32: output[i] = arg(0) >= arg(1); break;
					case Op::LogicalNot: output[i] = !arg(0); break;
					case Op::LogicalAnd: output[i] = arg(0) && arg(1); break;
					case Op::LogicalOr: output[i] = arg(0) || arg(1); break;
					case Op::LogicalXor: output[i] = static_cast<bool>(arg(0)) != static_cast<bool>(arg(1)); break;
					case Op::LoopGuardMark: result.marks += arg(0); break;
					default: throw std::logic_error("unsupported opcode in fixed loop test");
				}
			}
		}
	};
	const Block* pred = &entry;
	while (result.turns <= limit) {
		run(header, pred);
		run(test, &header);
		if (read(test.condition, 0) == 0) break;
		run(latch, &test);
		pred = &latch;
		++result.turns;
	}
	result.cursor = values.at(&cursor);
	result.payload = values.at(&payload);
	return result;
}

}

bool TestFixedLoopTail() {
	Inputs input {};
	input.start.fill(23);
	input.active.fill(1);
	for (const auto variant: {Variant::Plain, Variant::Coherent, Variant::Atomic, Variant::Barrier, Variant::TwoWaves}) {
		const auto r = Run(input, variant);
		const bool enabled = variant == Variant::Plain;
		if (r.rewritten != enabled || r.turns != (enabled ? 2u : 16u) || r.marks != 64u ||
		    !std::ranges::all_of(r.cursor, [](uint32_t i) { return i == 39u; })) return false;
	}
	for (const auto variant: {Variant::OuterLane, Variant::ScalarLane, Variant::OtherMask, Variant::EscapeStep, Variant::NoBallot}) {
		const auto r = Run(input, variant);
		if (!r.guarded || r.rewritten) return false;
	}
	input.start[63] = UINT32_MAX - 3u;
	const auto reference = Run(input, Variant::TwoWaves);
	const auto mixed = Run(input, Variant::Plain);
	if (mixed.turns >= reference.turns || mixed.cursor != reference.cursor || mixed.payload != reference.payload ||
	    mixed.marks != reference.marks || mixed.cursor[63] != 1u || mixed.payload[63] != 77u) return false;
	input.active.fill(0);
	input.active[63] = 1;
	const auto wrapped = Run(input, Variant::Plain);
	return wrapped.turns == 5u && wrapped.marks == 0u && wrapped.cursor[63] == 1u && wrapped.cursor[0] == 23u;
}
