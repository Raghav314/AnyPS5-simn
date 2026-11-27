#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t ShaderHeader = GuestBase + 0x100;
constexpr std::uint64_t ShaderCode = GuestBase + 0x800;
constexpr std::uint64_t Commands = GuestBase + 0x1000;
constexpr std::uint64_t Input = GuestBase + Block;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t GuestBytes = 3 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::string_view Notice = "bound as a null buffer";
constexpr std::string_view Abort = "invalid buffer discovery request";

constexpr std::array<std::uint32_t, 7> CopyWords{
    0x34020082, 0xe0301000, 0x80000201, 0xbf8c0000, 0xe0701000, 0x80010201, EndProgram,
};

using Sharp = std::array<std::uint32_t, 4>;

enum class Variant { Zero, Bound, Logged, Small, Straddle, RawFour, RawSix, RawSoffset, StructuredZero, StructuredNext,
                     IndexedZero, IndexedNext, IndexedWrap, NoStore, StoreZero, StoreNext, AtomicZero, AtomicNext };

struct Case {
    const char* name;
    Variant variant;
    bool notice;
};

constexpr std::array Cases{
    Case{"zero", Variant::Zero, false},
    Case{"bound", Variant::Bound, false},
    Case{"logged", Variant::Logged, true},
    Case{"small", Variant::Small, true},
    Case{"straddle", Variant::Straddle, true},
    Case{"raw-four", Variant::RawFour, false},
    Case{"raw-six", Variant::RawSix, false},
    Case{"raw-soffset", Variant::RawSoffset, false},
    Case{"structured-zero", Variant::StructuredZero, false},
    Case{"structured-next", Variant::StructuredNext, false},
    Case{"indexed-zero", Variant::IndexedZero, false},
    Case{"indexed-next", Variant::IndexedNext, false},
    Case{"indexed-wrap", Variant::IndexedWrap, false},
    Case{"no-store", Variant::NoStore, false},
    Case{"store-zero", Variant::StoreZero, false},
    Case{"store-next", Variant::StoreNext, false},
    Case{"atomic-zero", Variant::AtomicZero, false},
    Case{"atomic-next", Variant::AtomicNext, false},
};

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

Sharp sharp(std::uint64_t address, std::uint32_t records, std::uint32_t stride = 0) {
    const std::uint32_t select = stride != 0 ? 0u : 3u;
    return {static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records,
            (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (select << 28u)};
}

Sharp sourceSharp(Variant variant) {
    switch (variant) {
        case Variant::Zero:
        case Variant::Bound:
        case Variant::NoStore:
        case Variant::StoreZero:
        case Variant::StoreNext:
        case Variant::AtomicZero:
        case Variant::AtomicNext: return sharp(Input, Lanes * 4u);
        case Variant::Logged: return {0x402bd800u, 0x3ef38000u, 0x3e5e0000u, 0x00016204u};
        case Variant::Small: return {0x402bd800u, 0x00008000u, 0x3e5e0000u, 0x30016204u};
        case Variant::Straddle: return sharp((std::uint64_t{1} << 40u) - 0x1000u, 0x10000u);
        case Variant::RawFour: return sharp(Input, 4u);
        case Variant::RawSix: return sharp(Input, 6u);
        case Variant::RawSoffset: return sharp(Input, 4u);
        case Variant::StructuredZero:
        case Variant::IndexedZero:
        case Variant::IndexedWrap: return sharp(Input, Lanes, 4u);
        case Variant::StructuredNext:
        case Variant::IndexedNext: {
            auto descriptor = sharp(Input, Lanes + 1u, 4u);
            descriptor[3] |= 1u << 28u;
            return descriptor;
        }
    }
    throw std::runtime_error("unknown variant");
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords(Variant variant) {
    std::vector<std::uint32_t> words;
    const auto source = sourceSharp(variant);
    auto target = variant == Variant::NoStore || variant == Variant::StoreZero || variant == Variant::StoreNext ||
                  variant == Variant::AtomicZero || variant == Variant::AtomicNext
        ? sharp(Output, Lanes + (variant == Variant::StoreNext || variant == Variant::AtomicNext ? 1u : 0u), 4u)
        : sharp(Output, Width * Height * 4u);
    if (variant == Variant::StoreNext || variant == Variant::AtomicNext) target[3] |= 1u << 28u;
    append(words, 0x76, {0x20c, low(ShaderCode), high(ShaderCode)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, 8u << 1u});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]});
    append(words, 0x15, {1u, 1u, 1u, 0x41u});
    return words;
}

template <typename T, std::size_t Extent>
void write(std::span<std::byte> memory, std::uint64_t address, std::span<const T, Extent> items) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || items.size_bytes() > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, items.data(), items.size_bytes());
}

void writeShader(std::span<std::byte> memory, Variant variant) {
    std::vector<std::uint32_t> code(CopyWords.begin(), CopyWords.end());
    if (variant == Variant::RawSoffset) code[2] = 0x84000201u;
    if (variant == Variant::StructuredZero || variant == Variant::StructuredNext) code[1] |= 4u;
    if (variant == Variant::IndexedZero || variant == Variant::IndexedNext) {
        code[1] = 0xe0302004u;
        code[2] = 0x80000200u;
    }
    if (variant == Variant::IndexedWrap) {
        code.insert(code.begin() + 1, {0x38060081u, 0x3406069eu});
        code[3] = 0xe0302000u;
        code[4] = 0x80000203u;
    }
    if (variant == Variant::NoStore) code[4] = EndProgram;
    if (variant == Variant::StoreZero || variant == Variant::StoreNext) code[4] |= 4u;
    if (variant == Variant::AtomicZero || variant == Variant::AtomicNext) code[4] = 0xe0c81004u;
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size() * sizeof(std::uint32_t));
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, std::span<const Header>(&header, 1));
    write(memory, ShaderCode, std::span<const std::uint32_t>(code));
}

void makeCapture(const std::filesystem::path& path, Variant variant) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto commands = dispatchWords(variant);
    writeShader(memory, variant);
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> input(Lanes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        input[lane] = variant == Variant::Zero ? 0u : 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
    }
    write(memory, Input, std::span<const std::uint32_t>(input));
    std::vector<std::uint32_t> output(Width * Height, Fill);
    write(memory, Output, std::span<const std::uint32_t>(output));

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = ShaderHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Output,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":3,\"ranges\":1,"
                  "\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

struct Replay {
    bool completed = false;
    std::string log;
    std::string frame;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const Case& test) {
    const auto directory = root / test.name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", test.variant);
    const auto log = directory / "replay.log";
    const auto frames = directory / "frames";
    const auto command = quote(tool) + " " + quote(directory / "capture") +
                         " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(frames) + " > " +
                         quote(log) + " 2>&1";
    std::fflush(nullptr);
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
    Replay result;
    result.log = readFile(log);
    result.frame = readFile(frames / "replay-0.bmp");
    result.completed = status == 0 && !result.frame.empty();
    return result;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_buffer_null_range_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-buffer-null-range-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        std::string zero_frame;
        std::string bound_frame;
        std::string raw_four_frame;
        std::string no_store_frame;
        int failures = 0;
        for (const auto& test : Cases) {
            const auto result = replay(tool, root, test);
            if (test.variant == Variant::Zero) {
                if (!result.completed) {
                    std::cerr << "zero replay did not complete" << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)")
                              << '\n';
                    printTail(result.log);
                    return vulkan_required ? 1 : 77;
                }
                zero_frame = result.frame;
            }
            if (test.variant == Variant::Bound) bound_frame = result.frame;
            if (test.variant == Variant::RawFour) raw_four_frame = result.frame;
            if (test.variant == Variant::NoStore) no_store_frame = result.frame;
            std::string failure;
            if (!result.completed) {
                failure = result.log.find(Abort) != std::string::npos
                              ? "the replay stopped on an invalid buffer discovery request"
                              : "the replay failed";
            } else if (test.notice != (result.log.find(Notice) != std::string::npos)) {
                failure = test.notice ? "no null-buffer notice for a V# outside guest memory"
                                      : "unexpected null-buffer notice";
            } else if (test.variant == Variant::Bound && result.frame == zero_frame) {
                failure = "the bound V# did not copy its source";
            } else if (test.notice && result.frame != zero_frame) {
                failure = "a null buffer did not read as zero";
            } else if ((test.variant == Variant::RawFour || test.variant == Variant::StructuredNext ||
                        test.variant == Variant::IndexedNext) &&
                       result.frame == zero_frame) {
                failure = "an in-range buffer did not copy its source";
            } else if ((test.variant == Variant::RawSoffset || test.variant == Variant::StructuredZero ||
                        test.variant == Variant::IndexedZero || test.variant == Variant::IndexedWrap) &&
                       result.frame != zero_frame) {
                failure = "an out-of-range buffer did not read as zero";
            } else if (test.variant == Variant::RawSix && result.frame != raw_four_frame) {
                failure = "a dword straddling num_records did not read as zero";
            } else if ((test.variant == Variant::StoreZero || test.variant == Variant::AtomicZero) &&
                       result.frame != no_store_frame) {
                failure = "an out-of-range structured store changed output";
            } else if ((test.variant == Variant::StoreNext || test.variant == Variant::AtomicNext) &&
                       result.frame == no_store_frame) {
                failure = "an in-range structured store did not write";
            }
            if (failure.empty()) {
                std::cout << test.name << ": ok\n";
                continue;
            }
            ++failures;
            std::cerr << test.name << ": " << failure << '\n';
            printTail(result.log);
        }
        if (failures != 0) return 1;
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_buffer_null_range: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
