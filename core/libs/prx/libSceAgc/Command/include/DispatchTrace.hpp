#pragma once

#include <cstdint>
#include <cstdlib>
#include <string_view>

extern "C" void AgcDriverRecordComputePacketOrigin_nid_no_patch(const std::uint32_t* packet, std::uint32_t count,
                                                               std::uint64_t callerReturn);

namespace Agc::Command {

inline void TraceDispatch(const std::uint32_t* packet, std::uint32_t count, const void* callerReturn) {
    static const bool enabled = [] {
        const auto* value = std::getenv("KYTY_COMPUTE_DISPATCH_ORIGINS");
        return value != nullptr && std::string_view(value) == "1";
    }();
    if (enabled) {
        AgcDriverRecordComputePacketOrigin_nid_no_patch(packet, count, reinterpret_cast<std::uintptr_t>(callerReturn));
    }
}

}
