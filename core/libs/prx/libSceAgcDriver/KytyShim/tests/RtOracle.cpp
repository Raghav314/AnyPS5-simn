#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryResults.h"
#include "graphics/shader/recompiler/ir/passes/RayQuerySource.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <array>
#include <bit>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <spirv-tools/libspirv.hpp>
#include <vector>

namespace {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;
namespace CFG = Libs::Graphics::ShaderRecompiler::CFG;
namespace Spirv = Libs::Graphics::ShaderRecompiler::Spirv;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

enum class Node { Children, Uniform, Memory, Constant };
enum class Child { InLane, Ballot, ReadLane, Store };

struct Walk {
    IR::Block block;
    std::vector<IR::Inst*> sites;
};

IR::Inst& Intersection(IR::Block& block, IR::Value node, uint32_t node_words) {
    const IR::Value zero(uint32_t {0});
    const IR::Value one(uint32_t {0x3f800000u});
    auto& descriptor = block.AppendNewInst(IR::ValueOpcode::CompositeConstructU32x4, {zero, zero, zero, zero});
    IR::Value high = node_words == 2 ? IR::Value(uint32_t {0}) : one;
    auto& address = block.AppendNewInst(IR::ValueOpcode::MakeImageAddress,
                                        {node, high, one, zero, zero, zero, one, one, one, one, one, one, one});
    return block.AppendNewInst(IR::ValueOpcode::BvhIntersect, {IR::Value(&descriptor), IR::Value(&address), IR::Value(true)},
                               node_words);
}

void Build(Walk& walk, Node node, Child child, uint32_t node_words, uint32_t sites) {
    auto& block = walk.block;
    auto& phi = block.AppendNewInst(IR::ValueOpcode::Phi);
    phi.AddPhiOperand(&block, IR::Value(uint32_t {5}));
    IR::Value current(&phi);
    if (node == Node::Uniform) current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::ReadFirstLane, {current, IR::Value(true)}));
    if (node == Node::Memory) current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::LoadSharedU32, {current, IR::Value(true)}));
    if (node == Node::Constant) current = IR::Value(uint32_t {5});
    IR::Value last;
    for (uint32_t site = 0; site < sites; ++site) {
        auto& bvh = Intersection(block, current, node_words);
        walk.sites.push_back(&bvh);
        auto& first = block.AppendNewInst(IR::ValueOpcode::CompositeExtractU32x4, {IR::Value(&bvh), IR::Value(uint32_t {0})});
        auto& masked = block.AppendNewInst(IR::ValueOpcode::BitwiseAnd32, {IR::Value(&first), IR::Value(uint32_t {0xfffffff8u})});
        current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::BitwiseOr32, {IR::Value(&masked), IR::Value(uint32_t {5})}));
        last = IR::Value(&first);
    }
    auto& valid = block.AppendNewInst(IR::ValueOpcode::INotEqual32, {last, IR::Value(uint32_t {0xffffffffu})});
    switch (child) {
        case Child::InLane: break;
        case Child::Ballot: block.AppendNewInst(IR::ValueOpcode::Ballot, {IR::Value(&valid)}); break;
        case Child::ReadLane: block.AppendNewInst(IR::ValueOpcode::ReadLane, {last, IR::Value(uint32_t {0})}); break;
        case Child::Store: block.AppendNewInst(IR::ValueOpcode::WriteSharedU32, {IR::Value(uint32_t {0}), last, IR::Value(true)}); break;
    }
    if (node != Node::Constant) phi.AddPhiOperand(&block, current);
}

uint32_t Mark(Node node, Child child, uint32_t node_words = 1, uint32_t sites = 1) {
    Walk walk;
    Build(walk, node, child, node_words, sites);
    const auto flags = IR::MarkOracleWalk(walk.sites);
    for (const auto* bvh: walk.sites) {
        const auto set = bvh->Flags<uint32_t>();
        Expect((set & 0xffu) == node_words, "marking keeps the node word count");
        Expect((set & flags) == flags, "every intersection of the walk carries the walk flags");
    }
    return flags;
}

void EmptyWalks() {
    Walk served, left;
    Build(served, Node::Children, Child::InLane, 1, 1);
    Build(left, Node::Children, Child::InLane, 2, 2);
    IR::MarkOracleWalk(served.sites);
    std::vector<IR::Inst*> all(served.sites);
    all.insert(all.end(), left.sites.begin(), left.sites.end());
    IR::Program program;
    Expect(IR::BvhIntersectEmpty == 1u << 11, "the empty bit is bit 11");
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = false}) == 0, "the emulated variant runs no intersection empty");
    for (const auto* bvh: left.sites) {
        Expect((bvh->Flags<uint32_t>() & IR::BvhIntersectEmpty) == 0, "an emulated intersection keeps its box test");
    }
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = true}) == 2, "the native variant runs every unserved intersection empty");
    for (const auto* bvh: left.sites) {
        const auto set = bvh->Flags<uint32_t>();
        Expect((set & IR::BvhIntersectEmpty) != 0 && (set & IR::BvhIntersectNative) == 0, "an unserved intersection is empty, not native");
        Expect((set & 0xffu) == 2, "an empty intersection keeps its node word count");
    }
    for (const auto* bvh: served.sites) {
        Expect((bvh->Flags<uint32_t>() & IR::BvhIntersectEmpty) == 0, "an intersection the oracle serves is not empty");
    }
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = true}) == 0, "marking again adds nothing");
}

bool Lane(uint32_t flags) {
    return (flags & IR::BvhIntersectOracleLane) != 0;
}

void Marking() {
    const auto flags = Mark(Node::Children, Child::InLane);
    Expect((flags & IR::BvhIntersectNative) != 0 && (flags & IR::BvhIntersectOracle) != 0, "a marked walk is native and served by the oracle");
    Expect(IR::MarkOracleWalk({}) == 0u, "an empty walk marks nothing");
    Expect(IR::OracleRemaining == 1u && IR::OracleForms == 2u && IR::OracleOnly == 4u && IR::OraclePixel == 8u, "the oracle bits match KYTY_RT_ORACLE");
}

void Modes() {
    Expect(Lane(Mark(Node::Children, Child::InLane)), "a node fed only by the walk's own children walks per lane");
    Expect(Lane(Mark(Node::Children, Child::InLane, 2)), "a 64-bit node pointer fed by its children walks per lane");
    Expect(Lane(Mark(Node::Children, Child::InLane, 1, 2)), "two sites that feed each other walk per lane");
    Expect(!Lane(Mark(Node::Uniform, Child::InLane)), "a node read from the first lane walks the whole wave");
    Expect(!Lane(Mark(Node::Uniform, Child::InLane, 2)), "a 64-bit node read from the first lane walks the whole wave");
    Expect(!Lane(Mark(Node::Memory, Child::InLane)), "a node loaded from memory keeps the wave mode");
    Expect(!Lane(Mark(Node::Constant, Child::InLane)), "a constant node keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::Ballot)), "a child tested across lanes keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::ReadLane)), "a child read from another lane keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::Store)), "a child stored to memory keeps the wave mode");
}

void NodeCountFacts() {
    constexpr uint32_t Tlas = 1u << 29u;
    constexpr uint32_t Blas = 1u << 31u;
    IR::RayTracingLayout layout;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == 0, "both node counts at 88 by default");
    layout.node_count_80 = true;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == (Tlas | Blas), "an unset BLAS fact follows the TLAS one");
    layout.blas_node_count_80 = false;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == Tlas, "TLAS count at 80, BLAS count at 88");
    layout.node_count_80      = false;
    layout.blas_node_count_80 = true;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == Blas, "TLAS count at 88, BLAS count at 80");
    auto shaped = layout;
    shaped.blas_node_count_80.reset();
    Expect(shaped.Pack()[0] == layout.Pack()[0] && shaped.Pack()[1] == layout.Pack()[1] &&
               shaped.Pack()[2] == layout.Pack()[2] && ((shaped.Pack()[3] ^ layout.Pack()[3]) & ~Blas) == 0,
           "the BLAS fact changes bit 31 only");
}

void WaveScalarReadKindFacts() {
    IR::Program program;
    program.memory_info.push_back({.kind = IR::ResourceKind::ScalarBuffer, .offset = 0u});
    IR::Block block;
    const auto read = [&](IR::Value offset) -> IR::Inst& {
        return block.AppendNewInst(IR::ValueOpcode::ReadConstBuffer, {IR::Value(0u), offset});
    };
    auto& carried = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(3u), IR::Value(1u)});
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&carried))) == IR::RayQueryResultKind::Word,
           "a carry inside the register operand selects the second word");
    auto& dynamic = block.AppendNewInst(IR::ValueOpcode::GetUserData, {IR::Value(static_cast<IR::ScalarReg>(0u))});
    auto& unknown = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&dynamic), IR::Value(1u)});
    Expect(!IR::WaveScalarReadKind(program, read(IR::Value(&unknown))),
           "an unknown register residue cannot identify a wave result word");
    auto& aligned = block.AppendNewInst(IR::ValueOpcode::ShiftLeftLogical32, {IR::Value(&dynamic), IR::Value(3u)});
    auto& first = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&aligned), IR::Value(1u)});
    auto& second = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&aligned), IR::Value(4u)});
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&first))) == IR::RayQueryResultKind::Node,
           "a proven aligned register plus one selects the first word");
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&second))) == IR::RayQueryResultKind::Word,
           "a proven aligned register plus four selects the second word");
    auto& masked = block.AppendNewInst(IR::ValueOpcode::BitwiseAnd32, {IR::Value(&dynamic), IR::Value(~7u)});
    auto& masked_second = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&masked), IR::Value(4u)});
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&masked_second))) == IR::RayQueryResultKind::Word,
           "masking the register low bits proves the second word");
    auto& condition = block.AppendNewInst(IR::ValueOpcode::INotEqual32, {IR::Value(&dynamic), IR::Value(0u)});
    auto& selected = block.AppendNewInst(IR::ValueOpcode::SelectU32, {IR::Value(&condition), IR::Value(&aligned), IR::Value(5u)});
    Expect(!IR::WaveScalarReadKind(program, read(IR::Value(&selected))),
           "a selected aligned or unaligned register remains unclassified");
}

void TriangleIndexSpirv() {
    namespace Emit = Spirv::Emitter;
    const IR::Value unused(0u);
    IR::Block alpha;
    alpha.AppendNewInst(IR::ValueOpcode::RayQueryAlpha,
                        {unused, unused, unused, unused, unused, unused, unused,
                         unused, unused, unused, unused, unused, unused, unused});
    IR::Program program;
    program.blocks.push_back(&alpha);
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.lane_local = true;
    program.info.uses_dma = true;
    program.info.ray_query = true;
    const Libs::Graphics::ShaderComputeInputInfo compute {};
    const Spirv::Profile profile {.ray_query = true};
    Emit::EmitterState state(program, {.compute = &compute}, profile);
    Emit::ValueEmitContext ctx(state);
    Emit::DefineModule(state);
    Emit::DefineGetBdaPointer(state);
    Emit::DefineIndirectBufferLoads(ctx);
    auto& builder = state.builder;
    builder.AddFunction(spv::OpFunction, Emit::TypeVoid(state), state.main_func,
                        spv::FunctionControlMaskNone, Emit::TypeFunction(state));
    Emit::EmitLabel(state, state.entry_label);
    const auto zero = Emit::ConstantU32(state, 0);
    const auto indices = Emit::EmitTriangleIndices(ctx, {zero, zero, zero, zero}, zero);
    Expect(indices[0] != 0 && indices[1] != 0 && indices[2] != 0, "the triangle load emits three indices");
    builder.AddFunction(spv::OpReturn);
    builder.AddFunction(spv::OpFunctionEnd);
    Emit::DefineBdaLoads(state);
    builder.AddEntryPoint(spv::ExecutionModelGLCompute, state.main_func, "main", state.interface_variables);

    spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
    validator.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        std::fprintf(stderr, "%s\n", message);
    });
    Expect(validator.Validate(builder.Build()), "the ray-query triangle index load emits valid SPIR-V");
}

enum class HighWord { Exact, StrideBit, SwizzleBit };

struct SourceWalk {
    IR::Block              entry, header, body, latch, merge, second_header, second_body, second_latch, second_merge;
    IR::Program            program;
    IR::Block*             block = &entry;
    std::vector<IR::Inst*> sites;

    IR::Value Add(IR::ValueOpcode opcode, std::initializer_list<IR::Value> args, uint64_t flags = 0) {
        return IR::Value(&block->AppendNewInst(opcode, args, flags));
    }

    IR::Value Load(IR::ValueOpcode opcode, std::initializer_list<IR::Value> args, uint32_t offset, bool idxen) {
        IR::MemoryInfo memory;
        memory.offset = offset;
        memory.idxen  = idxen;
        program.memory_info.push_back(memory);
        const IR::MemoryFlags flags {static_cast<uint32_t>(program.memory_info.size() - 1), 0};
        return Add(opcode, args, std::bit_cast<uint64_t>(flags));
    }

    IR::Value LoadAddress(IR::Value resource, uint32_t offset) {
        return Load(IR::ValueOpcode::LoadAddressU32, {resource, IR::Value(uint32_t {0}), IR::Value(uint32_t {0}), IR::Value(true)}, offset, false);
    }

    std::array<IR::Value, 2> Offset(IR::Value low, IR::Value high, IR::Value node) {
        using Op         = IR::ValueOpcode;
        const auto sum   = Add(Op::IAdd32, {low, node});
        const auto carry = Add(Op::SelectU32, {Add(Op::ULessThan32, {sum, low}), IR::Value(uint32_t {1}), IR::Value(uint32_t {0})});
        return {sum, Add(Op::IAdd32, {high, carry})};
    }

    IR::Value Intersect(std::array<IR::Value, 2> node, const std::array<IR::Value, 6>& ray) {
        const IR::Value zero(uint32_t {0});
        const IR::Value one(uint32_t {0x3f800000u});
        const auto descriptor = Add(IR::ValueOpcode::CompositeConstructU32x4,
                                    {zero, IR::Value(uint32_t {0x80000000u}), IR::Value(uint32_t {0xffffffffu}), IR::Value(uint32_t {0x810003ffu})});
        const auto address = Add(IR::ValueOpcode::MakeImageAddress,
                                 {node[0], node[1], IR::Value(uint32_t {0x47c35000u}), ray[0], ray[1], ray[2], ray[3], ray[4], ray[5], one, one, one, zero});
        auto& bvh = block->AppendNewInst(IR::ValueOpcode::BvhIntersect, {descriptor, address, IR::Value(true)}, 2);
        sites.push_back(&bvh);
        return IR::Value(&bvh);
    }
};

void Link(IR::Block& from, IR::Block& to) {
    from.terminator.kind       = CFG::TerminatorKind::Branch;
    from.terminator.true_block = &to;
    from.AddBranch(&to);
}

struct Loop {
    IR::Block& before;
    IR::Block& header;
    IR::Block& body;
    IR::Block& latch;
    IR::Block& merge;
};

void Number(SourceWalk& walk) {
    walk.program.stage = Libs::Graphics::ShaderType::Compute;
    uint32_t pc        = 0;
    for (auto* block: walk.program.blocks) {
        block->start_pc = pc;
        block->end_pc   = pc += 16;
    }
}

std::array<IR::Value, 2> UserHeader(SourceWalk& walk, uint32_t offset) {
    using Op           = IR::ValueOpcode;
    walk.block         = &walk.entry;
    const auto user    = [&](uint32_t reg) { return walk.Add(Op::GetUserData, {IR::Value(IR::ScalarReg {reg})}); };
    const auto pointer = walk.Add(Op::GetAddressResource, {user(4), user(5)});
    return {walk.LoadAddress(pointer, offset), walk.LoadAddress(pointer, offset + 4)};
}

std::array<IR::Value, 6> UserRay(SourceWalk& walk) {
    walk.block = &walk.entry;
    std::array<IR::Value, 6> ray;
    for (uint32_t k = 0; k < 6; ++k) ray[k] = walk.Add(IR::ValueOpcode::GetUserData, {IR::Value(IR::ScalarReg {8 + k})});
    return ray;
}

void BuildWalk(SourceWalk& walk, const Loop& loop, std::array<IR::Value, 2> header, const std::array<IR::Value, 6>& ray, HighWord kind) {
    using Op               = IR::ValueOpcode;
    const auto u32         = [](uint32_t value) { return IR::Value(value); };
    const auto word        = static_cast<uint64_t>(IR::Type::U32);
    const auto header_low  = header[0];
    const auto header_high = header[1];
    walk.block             = &loop.before;
    const auto root        = walk.Add(Op::ShiftRightLogical64, {walk.Add(Op::CompositeConstructU64, {header_low, header_high}), u32(3)});
    const auto root_low    = walk.Add(Op::CompositeExtractU64, {root, u32(0)});
    const auto root_high   = walk.Add(Op::CompositeExtractU64, {root, u32(1)});
    Link(loop.before, loop.header);

    walk.block      = &loop.header;
    auto& base_low  = loop.header.AppendNewInst(Op::Phi, {}, word);
    auto& base_high = loop.header.AppendNewInst(Op::Phi, {}, word);
    auto& node      = loop.header.AppendNewInst(Op::Phi, {}, word);
    std::array<IR::Inst*, 6> rays {};
    std::array<IR::Value, 6> current;
    for (uint32_t k = 0; k < 6; ++k) {
        rays[k]    = &loop.header.AppendNewInst(Op::Phi, {}, word);
        current[k] = IR::Value(rays[k]);
    }
    Link(loop.header, loop.body);
    loop.header.terminator.merge_block    = &loop.merge;
    loop.header.terminator.continue_block = &loop.latch;
    loop.header.terminator.loop_header    = true;

    walk.block       = &loop.body;
    const auto bvh   = walk.Intersect(walk.Offset(IR::Value(&base_low), IR::Value(&base_high), IR::Value(&node)), current);
    const auto child = walk.Add(Op::CompositeExtractU32x4, {bvh, u32(0)});
    loop.body.condition              = walk.Add(Op::IEqual32, {child, u32(0xffffffffu)});
    loop.body.terminator.kind        = CFG::TerminatorKind::ConditionalBranch;
    loop.body.terminator.true_block  = &loop.merge;
    loop.body.terminator.false_block = &loop.latch;
    loop.body.AddBranch(&loop.merge);
    loop.body.AddBranch(&loop.latch);

    walk.block       = &loop.latch;
    const auto table = walk.Add(Op::GetAddressResource, {header_low, header_high});
    const auto low   = walk.Add(Op::IAdd32, {header_low, walk.LoadAddress(table, 0x30)});
    const auto carry = walk.Add(Op::SelectU32, {walk.Add(Op::ULessThan32, {low, header_low}), u32(1), u32(0)});
    auto       high  = walk.Add(Op::IAdd32, {walk.Add(Op::IAdd32, {header_high, walk.LoadAddress(table, 0x34)}), carry});
    if (kind == HighWord::StrideBit) high = walk.Add(Op::IAdd32, {high, u32(0x00010000u)});
    if (kind == HighWord::SwizzleBit) high = walk.Add(Op::BitwiseOr32, {high, u32(0x40000000u)});
    const auto buffer = walk.Add(Op::GetBufferResource, {low, walk.Add(Op::BitwiseOr32, {high, u32(0x00800000u)}), u32(0xffffffffu), u32(0x00027000u)});
    const auto index  = walk.Add(Op::ShiftRightLogical32, {child, u32(3)});
    const auto record = [&](uint32_t dword) {
        return walk.Load(Op::LoadBufferU32, {buffer, index, u32(0), u32(0), IR::Value(true)}, dword * 4u, true);
    };
    const auto number = [&](IR::Value value) { return walk.Add(Op::BitCastF32U32, {value}); };
    std::array<IR::Value, 6> local;
    for (uint32_t row = 0; row < 3; ++row) {
        std::array<IR::Value, 4> m;
        for (uint32_t c = 0; c < 4; ++c) m[c] = number(record(row * 4 + c));
        const auto origin =
            walk.Add(Op::FPMad32, {m[0], number(ray[0]), walk.Add(Op::FPMad32, {m[1], number(ray[1]), walk.Add(Op::FPMad32, {m[2], number(ray[2]), m[3]})})});
        const auto direction =
            walk.Add(Op::FPMad32, {m[0], number(ray[3]), walk.Add(Op::FPMad32, {m[1], number(ray[4]), walk.Add(Op::FPMul32, {m[2], number(ray[5])})})});
        local[row]     = walk.Add(Op::BitCastU32F32, {origin});
        local[row + 3] = walk.Add(Op::BitCastU32F32, {direction});
    }
    const auto leaf      = walk.Add(Op::IEqual32, {walk.Add(Op::BitwiseAnd32, {child, u32(7)}), u32(0)});
    const auto pick      = [&](IR::Value blas, IR::Value keep) { return walk.Add(Op::SelectU32, {leaf, blas, keep}); };
    const auto next_low  = pick(record(12), IR::Value(&base_low));
    const auto next_high = pick(walk.Add(Op::BitwiseAnd32, {record(13), u32(0x3fffffu)}), IR::Value(&base_high));
    const auto next_node = pick(record(14), child);
    Link(loop.latch, loop.header);
    base_low.AddPhiOperand(&loop.before, root_low);
    base_low.AddPhiOperand(&loop.latch, next_low);
    base_high.AddPhiOperand(&loop.before, root_high);
    base_high.AddPhiOperand(&loop.latch, next_high);
    node.AddPhiOperand(&loop.before, u32(5));
    node.AddPhiOperand(&loop.latch, next_node);
    for (uint32_t k = 0; k < 6; ++k) {
        rays[k]->AddPhiOperand(&loop.before, ray[k]);
        rays[k]->AddPhiOperand(&loop.latch, pick(local[k], current[k]));
    }
    loop.merge.AppendNewInst(Op::Phi, {}, word).AddPhiOperand(&loop.body, child);
}

IR::OracleWalks DeriveWithHighWord(HighWord kind) {
    SourceWalk walk;
    walk.program.blocks = {&walk.entry, &walk.header, &walk.body, &walk.latch, &walk.merge};
    Number(walk);
    const auto ray    = UserRay(walk);
    const auto header = UserHeader(walk, 0x20);
    BuildWalk(walk, {walk.entry, walk.header, walk.body, walk.latch, walk.merge}, header, ray, kind);
    return IR::DeriveOracleWalks(walk.program, walk.sites);
}

struct TwoWalks {
    SourceWalk             walk;
    IR::NativeTraversals   traced;
    std::array<uint32_t, 2> flags {};
};

void TraceTwoWalks(TwoWalks& two, uint32_t second_offset) {
    auto& walk          = two.walk;
    walk.program.blocks = {&walk.entry, &walk.header, &walk.body, &walk.latch, &walk.merge,
                           &walk.second_header, &walk.second_body, &walk.second_latch, &walk.second_merge};
    Number(walk);
    const auto ray    = UserRay(walk);
    const auto first  = UserHeader(walk, 0x20);
    const auto second = UserHeader(walk, second_offset);
    BuildWalk(walk, {walk.entry, walk.header, walk.body, walk.latch, walk.merge}, first, ray, HighWord::Exact);
    BuildWalk(walk, {walk.merge, walk.second_header, walk.second_body, walk.second_latch, walk.second_merge}, second, ray, HighWord::Exact);
    two.traced = IR::TraceNativeTraversals(walk.program, {.wave_size = 32, .rewrite = true, .oracle = IR::OracleRemaining | IR::OracleOnly});
    for (size_t at = 0; at < 2 && at < walk.sites.size(); ++at) two.flags[at] = walk.sites[at]->Flags<uint32_t>();
}

void TwoTlasWalks() {
    const auto multi = [](uint32_t flags) { return (flags & IR::BvhIntersectOracleMulti) != 0; };
    const auto host  = [](uint32_t flags) { return (flags & IR::BvhIntersectOracle) != 0; };
    TwoWalks   two;
    TraceTwoWalks(two, 0x40);
    Expect(two.traced.sites == 2 && host(two.flags[0]) && host(two.flags[1]), "both walks of a program over two TLASes are answered from the host query");
    Expect(two.traced.source_valid && two.traced.header_offset == 0x20u, "the first walk's TLAS is the primary source");
    Expect(two.traced.extra_sources.size() == 1 && two.traced.extra_sources.front().user_sgpr == 4u &&
               two.traced.extra_sources.front().header_offset == 0x40u,
           "the second walk's TLAS is bound as an extra source");
    Expect(!multi(two.flags[0]), "the walk over the primary TLAS keeps the single-TLAS query");
    Expect(multi(two.flags[1]), "the walk over the other TLAS selects its TLAS per lane");
    TwoWalks same;
    TraceTwoWalks(same, 0x20);
    Expect(same.traced.sites == 2 && same.traced.extra_sources.empty() && !multi(same.flags[0]) && !multi(same.flags[1]),
           "two walks over one TLAS bind no extra source and keep the single-TLAS query");
}

void RecordLayouts() {
    constexpr std::array<uint32_t, 4> masked {0x76543210u, 0x3dc6ba98u, 0x000801dfu, 0u};
    constexpr std::array<uint32_t, 4> plain {0x76543210u, 0x3dc6ba98u, 0u, 0u};
    constexpr std::array<uint32_t, 4> reordered {0x76543201u, 0x3dc6ba98u, 0u, 0u};
    Expect(IR::CompatibleRecordLayouts(masked, masked), "a record layout is compatible with itself");
    Expect(IR::CompatibleRecordLayouts(masked, plain) && IR::CompatibleRecordLayouts(plain, masked),
           "layouts that differ only in the mask fields read the same records");
    Expect(!IR::CompatibleRecordLayouts(plain, reordered), "layouts that place record fields differently are not compatible");
}

void DescriptorHighWord() {
    constexpr const char* unproven = "a buffer descriptor's high word is not proven to leave the stride bits clear";
    const auto            refused  = [&](HighWord kind) {
        const auto walks = DeriveWithHighWord(kind);
        return walks.refusal != nullptr && std::strcmp(walks.refusal, unproven) == 0;
    };
    const auto exact = DeriveWithHighWord(HighWord::Exact);
    Expect(exact.refusal == nullptr && !exact.sources.empty(), "a walk whose descriptor high word is the high dword of header plus instance offset is derived");
    Expect(!exact.sources.empty() && exact.sources.front().user_sgpr == 4u && exact.sources.front().header_offset == 0x20u,
           "the derived walk reads the TLAS header the user data names");
    Expect(refused(HighWord::StrideBit), "a computed high word that sets a stride bit is refused");
    Expect(refused(HighWord::SwizzleBit), "a computed high word that sets a swizzle bit is refused");
}

void PrologueFlag() {
    using Op = IR::ValueOpcode;
    IR::Block   entry, header, body, merge;
    IR::Program program;
    program.blocks = {&entry, &header, &body, &merge};
    program.stage  = Libs::Graphics::ShaderType::Compute;
    const auto u32  = [](uint32_t value) { return IR::Value(value); };
    const auto flag = IR::Value(&entry.AppendNewInst(Op::INotEqual32, {IR::Value(&entry.AppendNewInst(Op::LaneId)), u32(2)}));
    entry.terminator.kind       = CFG::TerminatorKind::Branch;
    entry.terminator.true_block = &header;
    entry.AddBranch(&header);
    const auto word  = static_cast<uint64_t>(IR::Type::U32);
    auto& node       = header.AppendNewInst(Op::Phi, {}, word);
    auto& descriptor = header.AppendNewInst(Op::CompositeConstructU32x4, {u32(0), u32(0), u32(0), u32(0)});
    const IR::Value one(uint32_t {0x3f800000u});
    auto& address = header.AppendNewInst(Op::MakeImageAddress, {IR::Value(&node), one, one, u32(0), u32(0), u32(0), one, one, one, one, one, one, one});
    auto& bvh     = header.AppendNewInst(Op::BvhIntersect, {IR::Value(&descriptor), IR::Value(&address), flag}, 1);
    auto& first   = header.AppendNewInst(Op::CompositeExtractU32x4, {IR::Value(&bvh), u32(0)});
    header.condition              = IR::Value(&header.AppendNewInst(Op::IEqual32, {IR::Value(&first), u32(0xffffffffu)}));
    header.terminator.kind        = CFG::TerminatorKind::ConditionalBranch;
    header.terminator.true_block  = &merge;
    header.terminator.false_block = &body;
    header.terminator.merge_block = &merge;
    header.terminator.loop_header = true;
    header.AddBranch(&merge);
    header.AddBranch(&body);
    body.terminator.kind       = CFG::TerminatorKind::Branch;
    body.terminator.true_block = &header;
    body.AddBranch(&header);
    node.AddPhiOperand(&entry, u32(5));
    node.AddPhiOperand(&body, IR::Value(&first));
    merge.AppendNewInst(Op::Phi, {}, word).AddPhiOperand(&header, IR::Value(&first));
    std::vector<IR::Inst*> served {&bvh};
    const auto sample = IR::SampleWalkEntry(program, {.wave_size = 32}, &header, &body, served, 1);
    Expect(sample.failure == nullptr, "the prologue of the test walk runs");
    if (sample.failure != nullptr || sample.flags.size() != 1 || sample.active.size() != 32) {
        Expect(false, "the test walk samples one entry flag per lane");
        return;
    }
    const auto& lanes = sample.flags.front();
    Expect(sample.active[2] != 0 && lanes[2] == 0u, "an active lane whose prologue computes false enters the walk disabled");
    Expect(sample.active[0] != 0 && lanes[0] == 1u, "an active lane whose prologue computes true enters the walk enabled");
    for (uint32_t lane = 0; lane < 32; ++lane) {
        if (sample.active[lane] == 0) Expect(lanes[lane] == 0u, "an inactive lane enters the walk disabled");
    }
}

void EntryLanes() {
    Expect(IR::EntryFlagOf(0u, true) == 0u, "a lane the prologue disables stays disabled while active");
    Expect(IR::EntryFlagOf(1u, true) == 1u, "a lane the prologue enables stays enabled while active");
    Expect(IR::EntryFlagOf(1u, false) == 0u, "an inactive lane is disabled");
    Expect(IR::EntryMaskOf(0b0101u, 0b0111u) == 0b0101u, "a mask the prologue narrows is not widened to the active lanes");
    Expect(IR::EntryMaskOf(0b1111u, 0b0011u) == 0b0011u, "a mask is narrowed to the active lanes");
}

}

int main() {
    EntryLanes();
    PrologueFlag();
    TwoTlasWalks();
    RecordLayouts();
    DescriptorHighWord();
    Marking();
    EmptyWalks();
    Modes();
    NodeCountFacts();
    WaveScalarReadKindFacts();
    TriangleIndexSpirv();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim ray tracing oracle: all checks passed\n");
    return 0;
}
