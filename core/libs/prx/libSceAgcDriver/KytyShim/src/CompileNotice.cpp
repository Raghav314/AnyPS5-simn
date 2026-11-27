#include "prx/libSceAgcDriver/KytyShim/src/CompileNotice.hpp"

#include "graphics/presentation/precompileNotice.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace KytyShim::CompileNotice {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::int32_t GlyphWidth = 5;
constexpr std::int32_t GlyphHeight = 9;
constexpr std::int32_t Advance = 6;
constexpr std::int32_t LinePitch = 12;
constexpr std::int32_t Padding = 4;
constexpr std::int32_t Margin = 8;
constexpr std::int32_t BarHeight = 5;
constexpr std::int32_t CompactColumns = 46;
constexpr std::int32_t ScreenColumns = 46;
constexpr std::int32_t CompactLinesPerScale = 270;
constexpr std::int32_t ScreenLinesPerScale = 180;
constexpr auto RateWindow = std::chrono::seconds(10);
constexpr auto EstimatePeriod = std::chrono::seconds(1);
constexpr auto EstimateSpan = std::chrono::seconds(2);
constexpr auto WaitingLinger = std::chrono::seconds(2);

constexpr std::array<std::array<std::uint8_t, GlyphHeight>, 95> Glyphs{{
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04},
    {0x00, 0x00, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a},
    {0x00, 0x00, 0x04, 0x0f, 0x14, 0x0e, 0x05, 0x1e, 0x04},
    {0x00, 0x00, 0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03},
    {0x00, 0x00, 0x0c, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0d},
    {0x00, 0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02},
    {0x00, 0x00, 0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08},
    {0x00, 0x00, 0x00, 0x04, 0x15, 0x0e, 0x15, 0x04, 0x00},
    {0x00, 0x00, 0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x04, 0x08},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c},
    {0x00, 0x00, 0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00},
    {0x00, 0x00, 0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e},
    {0x00, 0x00, 0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e},
    {0x00, 0x00, 0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f},
    {0x00, 0x00, 0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e},
    {0x00, 0x00, 0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02},
    {0x00, 0x00, 0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e},
    {0x00, 0x00, 0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e},
    {0x00, 0x00, 0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e},
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c},
    {0x00, 0x00, 0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x0c, 0x00},
    {0x00, 0x00, 0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x04, 0x08},
    {0x00, 0x00, 0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02},
    {0x00, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00},
    {0x00, 0x00, 0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08},
    {0x00, 0x00, 0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04},
    {0x00, 0x00, 0x0e, 0x11, 0x01, 0x0d, 0x15, 0x15, 0x0e},
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11},
    {0x00, 0x00, 0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e},
    {0x00, 0x00, 0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e},
    {0x00, 0x00, 0x1c, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1c},
    {0x00, 0x00, 0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f},
    {0x00, 0x00, 0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10},
    {0x00, 0x00, 0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f},
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11},
    {0x00, 0x00, 0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e},
    {0x00, 0x00, 0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c},
    {0x00, 0x00, 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
    {0x00, 0x00, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f},
    {0x00, 0x00, 0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11},
    {0x00, 0x00, 0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11},
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},
    {0x00, 0x00, 0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10},
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d},
    {0x00, 0x00, 0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11},
    {0x00, 0x00, 0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e},
    {0x00, 0x00, 0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04},
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a},
    {0x00, 0x00, 0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11},
    {0x00, 0x00, 0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04},
    {0x00, 0x00, 0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f},
    {0x00, 0x00, 0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e},
    {0x00, 0x00, 0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00},
    {0x00, 0x00, 0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e},
    {0x00, 0x00, 0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f},
    {0x00, 0x00, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x0e, 0x01, 0x0f, 0x11, 0x0f},
    {0x00, 0x00, 0x10, 0x10, 0x1e, 0x11, 0x11, 0x11, 0x1e},
    {0x00, 0x00, 0x00, 0x00, 0x0e, 0x10, 0x10, 0x11, 0x0e},
    {0x00, 0x00, 0x01, 0x01, 0x0f, 0x11, 0x11, 0x11, 0x0f},
    {0x00, 0x00, 0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0e},
    {0x00, 0x00, 0x06, 0x09, 0x08, 0x1c, 0x08, 0x08, 0x08},
    {0x00, 0x00, 0x00, 0x00, 0x0f, 0x11, 0x0f, 0x01, 0x0e},
    {0x00, 0x00, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11},
    {0x00, 0x00, 0x04, 0x00, 0x0c, 0x04, 0x04, 0x04, 0x0e},
    {0x00, 0x00, 0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0c},
    {0x00, 0x00, 0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12},
    {0x00, 0x00, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e},
    {0x00, 0x00, 0x00, 0x00, 0x1a, 0x15, 0x15, 0x11, 0x11},
    {0x00, 0x00, 0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11},
    {0x00, 0x00, 0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e},
    {0x00, 0x00, 0x00, 0x00, 0x1e, 0x11, 0x1e, 0x10, 0x10},
    {0x00, 0x00, 0x00, 0x00, 0x0d, 0x13, 0x0f, 0x01, 0x01},
    {0x00, 0x00, 0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10},
    {0x00, 0x00, 0x00, 0x00, 0x0e, 0x10, 0x0e, 0x01, 0x1e},
    {0x00, 0x00, 0x08, 0x08, 0x1c, 0x08, 0x08, 0x09, 0x06},
    {0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d},
    {0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04},
    {0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a},
    {0x00, 0x00, 0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11},
    {0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x0f, 0x01, 0x0e},
    {0x00, 0x00, 0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f},
    {0x00, 0x00, 0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02},
    {0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
    {0x00, 0x00, 0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08},
    {0x00, 0x00, 0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00},
}};

struct Notice {
    Libs::Graphics::PrecompileNotice data;
    Clock::time_point since;
    std::uint64_t waitingHash = 0;
    std::uint32_t waitingSeconds = 0;
    Clock::time_point waitingAt;
};

struct Manifest {
    Libs::Graphics::ManifestProgress data;
    Clock::time_point since;
    std::deque<std::pair<Clock::time_point, std::uint32_t>> samples;
    std::optional<double> remaining;
    Clock::time_point estimated;
};

struct NoticeState {
    std::mutex mutex;
    std::optional<Notice> notice;
    std::optional<Manifest> manifest;
    bool manifestHidden = false;
};

NoticeState& State() {
    static auto* state = new NoticeState;
    return *state;
}

std::atomic<std::uint64_t> visualRevision{0};

void PublishVisibility(const NoticeState& state) {
    const bool visible = state.notice.has_value() || state.manifest.has_value();
    const auto revision = visualRevision.load(std::memory_order_relaxed);
    if (visible != ((revision & 1u) != 0)) visualRevision.store(revision + 1, std::memory_order_release);
}

void Estimate(Manifest& manifest, Clock::time_point now) {
    manifest.samples.emplace_back(now, manifest.data.done);
    while (manifest.samples.size() > 2 && now - manifest.samples.front().first > RateWindow) manifest.samples.pop_front();
    if (now - manifest.estimated < EstimatePeriod) return;
    manifest.estimated = now;
    const auto total = std::max(manifest.data.total, manifest.data.done);
    if (manifest.data.done >= total) {
        manifest.remaining = 0.0;
        return;
    }
    const auto& [oldestAt, oldestDone] = manifest.samples.front();
    const auto span = std::chrono::duration<double>(now - oldestAt).count();
    if (now - oldestAt < EstimateSpan || manifest.data.done <= oldestDone) {
        if (now - oldestAt >= EstimateSpan) manifest.remaining.reset();
        return;
    }
    const auto rate = static_cast<double>(manifest.data.done - oldestDone) / span;
    manifest.remaining = static_cast<double>(total - manifest.data.done) / rate;
}

std::string Duration(double seconds) {
    const auto whole = static_cast<std::uint64_t>(std::max(0.0, seconds) + 0.5);
    if (whole < 60) return std::to_string(whole) + " s";
    const auto rest = whole % 60;
    return std::to_string(whole / 60) + " min " + (rest < 10 ? "0" : "") + std::to_string(rest) + " s";
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%016" PRIx64, value);
    return text;
}

void Glyph(AgcDriver::OverlayLayout& layout, char character, std::int32_t left, std::int32_t top, std::int32_t scale, AgcDriver::OverlayColor color) {
    const auto index = character >= 0x20 && character <= 0x7E ? static_cast<std::size_t>(character - 0x20) : static_cast<std::size_t>('?' - 0x20);
    const auto& rows = Glyphs[index];
    for (std::int32_t y = 0; y < GlyphHeight; ++y) {
        const auto bits = rows[static_cast<std::size_t>(y)];
        std::int32_t x = 0;
        while (x < GlyphWidth) {
            if ((bits & (0x10u >> x)) == 0) {
                ++x;
                continue;
            }
            auto end = x;
            while (end < GlyphWidth && (bits & (0x10u >> end)) != 0) ++end;
            layout.rects.push_back({left + x * scale, top + y * scale, (end - x) * scale, scale, static_cast<std::uint32_t>(color)});
            x = end;
        }
    }
}

std::int32_t TextWidth(const std::string& text) {
    return text.empty() ? 0 : static_cast<std::int32_t>(text.size()) * Advance - 1;
}

void Text(AgcDriver::OverlayLayout& layout, const std::string& text, std::int32_t left, std::int32_t top, std::int32_t scale, AgcDriver::OverlayColor color) {
    for (std::size_t index = 0; index < text.size(); ++index) Glyph(layout, text[index], left + static_cast<std::int32_t>(index) * Advance * scale, top, scale, color);
}

void Rect(AgcDriver::OverlayLayout& layout, std::int32_t left, std::int32_t top, std::int32_t width, std::int32_t height, AgcDriver::OverlayColor color) {
    if (width > 0 && height > 0) layout.rects.push_back({left, top, width, height, static_cast<std::uint32_t>(color)});
}

struct Snapshot {
    std::uint32_t done = 0;
    std::uint32_t total = 0;
    double elapsed = 0.0;
    bool manifest = false;
    std::uint64_t cached = 0;
    std::uint64_t translated = 0;
    std::uint64_t compiled = 0;
    std::optional<double> remaining;
    std::uint64_t waitingHash = 0;
    std::uint32_t waitingSeconds = 0;
    bool boot = false;
};

std::optional<Snapshot> Take() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (!state.notice.has_value() && !state.manifest.has_value()) return std::nullopt;
    const auto now = Clock::now();
    Snapshot shot;
    if (state.manifest.has_value()) {
        const auto& manifest = *state.manifest;
        shot.manifest = true;
        shot.done = manifest.data.done;
        shot.total = std::max(manifest.data.total, manifest.data.done);
        shot.elapsed = std::chrono::duration<double>(now - manifest.since).count();
        shot.cached = manifest.data.cached;
        shot.translated = manifest.data.translated;
        shot.compiled = manifest.data.compiled;
        shot.remaining = manifest.remaining;
        shot.boot = !state.manifestHidden;
    }
    if (state.notice.has_value()) {
        const auto& notice = *state.notice;
        if (!shot.manifest) {
            shot.done = notice.data.done;
            shot.total = std::max(notice.data.total, notice.data.done);
            shot.elapsed = std::chrono::duration<double>(now - notice.since).count();
        }
        if (notice.waitingHash != 0 && now - notice.waitingAt < WaitingLinger) {
            shot.waitingHash = notice.waitingHash;
            shot.waitingSeconds = notice.waitingSeconds;
        }
    }
    return shot;
}

std::string Counts(const Snapshot& shot) {
    return std::to_string(shot.done) + " / " + std::to_string(shot.total);
}

std::string Waiting(const Snapshot& shot) {
    if (shot.waitingHash == 0) return {};
    return std::string("Waiting for shader ") + Hex(shot.waitingHash) + " (" + std::to_string(shot.waitingSeconds) + " s)";
}

std::string Remaining(const Snapshot& shot) {
    if (!shot.remaining.has_value()) return "estimating";
    return "about " + Duration(*shot.remaining);
}

std::int32_t Scale(std::int32_t height, std::int32_t perScale) {
    return std::max<std::int32_t>(2, height / perScale);
}

void Bar(AgcDriver::OverlayLayout& layout, const Snapshot& shot, std::int32_t left, std::int32_t top, std::int32_t width, std::int32_t height) {
    Rect(layout, left, top, width, height, AgcDriver::OverlayColor::Key);
    if (shot.total == 0) return;
    const auto filled = static_cast<std::int32_t>(static_cast<std::int64_t>(width) * shot.done / shot.total);
    Rect(layout, left, top, filled, height, AgcDriver::OverlayColor::Graph);
}

void LayOutScreen(AgcDriver::OverlayLayout& layout, const Snapshot& shot, std::int32_t width, std::int32_t height) {
    std::vector<std::string> lines;
    const auto percent = shot.total != 0 ? static_cast<std::uint64_t>(shot.done) * 100 / shot.total : 0;
    lines.push_back(shot.total != 0 ? Counts(shot) + "  (" + std::to_string(percent) + " %)" : std::string());
    lines.push_back(shot.manifest ? "Shaders loaded from cache: " + std::to_string(shot.cached) : std::string());
    lines.push_back(shot.manifest ? "Shaders translated: " + std::to_string(shot.translated) : std::string());
    lines.push_back(shot.manifest ? "Pipelines compiled by the driver: " + std::to_string(shot.compiled) : std::string());
    lines.push_back("Elapsed: " + Duration(shot.elapsed) + (shot.manifest ? "   Remaining: " + Remaining(shot) : std::string()));
    lines.push_back(Waiting(shot));
    const std::string title = "Compiling shaders";
    const auto contentWidth = ScreenColumns * Advance - 1;
    const auto panelWidth = contentWidth + 2 * Padding;
    const auto titleHeight = GlyphHeight * 2;
    const auto panelHeight = 2 * Padding + titleHeight + Padding + BarHeight + Padding + static_cast<std::int32_t>(lines.size()) * LinePitch;
    auto scale = Scale(height, ScreenLinesPerScale);
    while (scale > 1 && ((panelWidth + 2 * Margin) * scale > width || (panelHeight + 2 * Margin) * scale > height)) --scale;
    const auto left = (width - panelWidth * scale) / 2;
    const auto top = (height - panelHeight * scale) / 2;
    Rect(layout, left, top, panelWidth * scale, panelHeight * scale, AgcDriver::OverlayColor::Panel);
    Rect(layout, left, top, panelWidth * scale, scale, AgcDriver::OverlayColor::Accent);
    const auto contentLeft = left + Padding * scale;
    auto y = top + Padding * scale;
    Text(layout, title, left + (panelWidth * scale - TextWidth(title) * scale * 2) / 2, y, scale * 2, AgcDriver::OverlayColor::Accent);
    y += (titleHeight + Padding) * scale;
    Bar(layout, shot, contentLeft, y, contentWidth * scale, BarHeight * scale);
    y += (BarHeight + Padding) * scale;
    for (const auto& line : lines) {
        Text(layout, line, contentLeft, y, scale, AgcDriver::OverlayColor::Text);
        y += LinePitch * scale;
    }
}

void LayOutCompact(AgcDriver::OverlayLayout& layout, const Snapshot& shot, std::int32_t width, std::int32_t height) {
    std::vector<std::string> lines;
    std::string first = "Compiling shaders";
    if (shot.total != 0) first += ": " + Counts(shot);
    first += " (" + Duration(shot.elapsed) + ")";
    lines.push_back(first);
    if (shot.manifest) lines.push_back("Remaining: " + Remaining(shot));
    if (const auto waiting = Waiting(shot); !waiting.empty()) lines.push_back(waiting);
    const auto count = static_cast<std::int32_t>(lines.size());
    const auto contentWidth = CompactColumns * Advance - 1;
    const auto panelWidth = contentWidth + 2 * Padding;
    const auto bar = shot.manifest ? Padding + BarHeight : 0;
    const auto panelHeight = 2 * Padding + count * GlyphHeight + (count - 1) * (LinePitch - GlyphHeight) + bar;
    auto scale = Scale(height, CompactLinesPerScale);
    while (scale > 1 && ((panelWidth + 2 * Margin) * scale > width || (panelHeight + 2 * Margin) * scale > height)) --scale;
    const auto left = Margin * scale;
    const auto top = height - (Margin + panelHeight) * scale;
    Rect(layout, left, top, panelWidth * scale, panelHeight * scale, AgcDriver::OverlayColor::Panel);
    const auto contentLeft = left + Padding * scale;
    auto y = top + Padding * scale;
    for (std::int32_t index = 0; index < count; ++index) {
        if (index != 0) y += (LinePitch - GlyphHeight) * scale;
        Text(layout, lines[static_cast<std::size_t>(index)], contentLeft, y, scale, index == 0 ? AgcDriver::OverlayColor::Accent : AgcDriver::OverlayColor::Text);
        y += GlyphHeight * scale;
        if (index == 0 && shot.manifest) {
            y += Padding * scale;
            Bar(layout, shot, contentLeft, y, contentWidth * scale, BarHeight * scale);
            y += BarHeight * scale;
        }
    }
}

}

std::uint64_t Revision() noexcept {
    return visualRevision.load(std::memory_order_acquire);
}

bool LayOut(AgcDriver::OverlayLayout& layout, VkExtent2D extent, bool loading) {
    if (extent.width == 0 || extent.height == 0) return false;
    const auto shot = Take();
    if (!shot.has_value()) return false;
    const auto width = static_cast<std::int32_t>(extent.width);
    const auto height = static_cast<std::int32_t>(extent.height);
    if (loading || shot->boot) LayOutScreen(layout, *shot, width, height);
    else LayOutCompact(layout, *shot, width, height);
    return true;
}

}

namespace Libs::Graphics {

bool ManifestProgressVisible() noexcept {
    auto& state = KytyShim::CompileNotice::State();
    std::lock_guard lock(state.mutex);
    return state.manifest.has_value();
}

void ShowManifestProgress(const ManifestProgress& progress) {
    auto& state = KytyShim::CompileNotice::State();
    std::lock_guard lock(state.mutex);
    const auto now = KytyShim::CompileNotice::Clock::now();
    if (!state.manifest.has_value()) state.manifest = KytyShim::CompileNotice::Manifest{progress, now, {}, std::nullopt, now};
    state.manifest->data = progress;
    KytyShim::CompileNotice::Estimate(*state.manifest, now);
    KytyShim::CompileNotice::PublishVisibility(state);
}

void HideManifestProgress() {
    auto& state = KytyShim::CompileNotice::State();
    std::lock_guard lock(state.mutex);
    if (state.manifest.has_value()) state.manifestHidden = true;
    state.manifest.reset();
    KytyShim::CompileNotice::PublishVisibility(state);
}

void ShowPrecompileNotice(const PrecompileNotice& notice) {
    auto& state = KytyShim::CompileNotice::State();
    std::lock_guard lock(state.mutex);
    if (!notice.visible) {
        state.notice.reset();
        KytyShim::CompileNotice::PublishVisibility(state);
        return;
    }
    const auto now = KytyShim::CompileNotice::Clock::now();
    if (!state.notice.has_value()) state.notice = KytyShim::CompileNotice::Notice{notice, now};
    state.notice->data = notice;
    KytyShim::CompileNotice::PublishVisibility(state);
    if (notice.waiting_hash != 0) {
        state.notice->waitingHash = notice.waiting_hash;
        state.notice->waitingSeconds = notice.waiting_seconds;
        state.notice->waitingAt = now;
    }
}

void HidePrecompileNotice() {
    auto& state = KytyShim::CompileNotice::State();
    std::lock_guard lock(state.mutex);
    state.notice.reset();
    KytyShim::CompileNotice::PublishVisibility(state);
}

}
