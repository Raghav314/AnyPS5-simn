#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics {

struct ComputeDispatchOrigin {
	uint64_t packet = 0;
	uint64_t caller_return = 0;
	uint64_t submit = 0;
};

namespace ComputeDispatchTrace {

inline bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_COMPUTE_DISPATCH_ORIGINS");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

struct PacketOrigin {
	uint64_t packet = 0;
	uint64_t caller_return = 0;
	std::array<uint32_t, 5> words {};
	uint32_t count = 0;
};

inline constexpr size_t PacketSlots = 65536;
inline std::array<PacketOrigin, PacketSlots> packet_origins;
inline std::mutex packet_mutex;
inline std::unordered_set<uint64_t> reported_variants;
inline thread_local ComputeDispatchOrigin current;

inline size_t Slot(uint64_t packet) {
	return static_cast<size_t>(((packet >> 2) * 0x9e3779b97f4a7c15ull) >> 48);
}

inline void Record(const uint32_t* packet, uint32_t count, uint64_t caller_return) {
	if (!Enabled() || packet == nullptr || count < 3 || count > 5) return;
	const auto address = reinterpret_cast<uint64_t>(packet);
	std::lock_guard lock(packet_mutex);
	auto& origin = packet_origins[Slot(address)];
	origin.packet = address;
	origin.caller_return = caller_return;
	origin.count = count;
	std::copy_n(packet, count, origin.words.begin());
}

inline ComputeDispatchOrigin Capture(std::span<const uint32_t> packet, uint64_t submit) {
	if (!Enabled() || packet.empty()) return {};
	const auto address = reinterpret_cast<uint64_t>(packet.data());
	std::lock_guard lock(packet_mutex);
	const auto& origin = packet_origins[Slot(address)];
	const bool match = origin.packet == address && origin.count == packet.size() &&
	                   std::ranges::equal(packet, std::span(origin.words).first(origin.count));
	return {address, match ? origin.caller_return : 0, submit};
}

class Scope {
public:
	explicit Scope(ComputeDispatchOrigin origin): previous(std::exchange(current, origin)) {}
	~Scope() { current = previous; }
	Scope(const Scope&) = delete;
	Scope& operator=(const Scope&) = delete;

private:
	ComputeDispatchOrigin previous;
};

inline bool FirstVariant(uint64_t variant) {
	if (current.packet == 0) return false;
	std::lock_guard lock(packet_mutex);
	return reported_variants.insert(variant).second;
}

}

}
