#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/presentation/precompileNotice.h"
#include "graphics/presentation/window.h"

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <span>
#include <stop_token>
#include <vector>

#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace {
std::string g_title;
std::string g_version;
PFN_vkCreateComputePipelines g_create_compute_pipelines = nullptr;
std::mutex g_jobs_mutex;
std::condition_variable g_jobs_available;
unsigned g_jobs_active = 0;
unsigned g_jobs_peak = 0;
unsigned g_jobs_limit = 4;
uint64_t g_finished_pipelines = 0;
const char* g_four_job_marker = nullptr;

struct DriverComputeTiming {
	uint64_t calls = 0;
	uint64_t successful = 0;
	std::chrono::steady_clock::duration waited {};
	std::chrono::steady_clock::duration duration {};
	std::chrono::steady_clock::duration longest {};

	void Record(VkResult result, std::chrono::steady_clock::duration queue_time,
	            std::chrono::steady_clock::duration driver_time) {
		calls++;
		successful += result == VK_SUCCESS;
		waited += queue_time;
		duration += driver_time;
		longest = std::max(longest, driver_time);
	}

	void Report(const char* route) const {
		const auto millis = [](auto elapsed) { return std::chrono::duration<double, std::milli>(elapsed).count(); };
		std::fprintf(stderr,
		             "kyty_precache: compute_%s calls=%llu success=%llu queue_ms=%.1f driver_ms=%.1f longest_ms=%.1f\n",
		             route, static_cast<unsigned long long>(calls), static_cast<unsigned long long>(successful),
		             millis(waited), millis(duration), millis(longest));
	}
};

DriverComputeTiming g_lookup_time;
DriverComputeTiming g_create_time;

unsigned DriverComputeLimit() {
	const auto fallback = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 4u);
	const auto* value = std::getenv("KYTY_PRECACHE_DRIVER_JOBS");
	return value == nullptr ? fallback : static_cast<unsigned>(std::clamp(std::atoi(value), 1, 6));
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                      const VkComputePipelineCreateInfo* infos,
                                                      const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
	const auto asked = std::chrono::steady_clock::now();
	{
		std::unique_lock lock(g_jobs_mutex);
		g_jobs_available.wait(lock, [] {
			if (g_four_job_marker && g_jobs_limit > 4 && access(g_four_job_marker, F_OK) == 0) g_jobs_limit = 4;
			return g_jobs_active < g_jobs_limit;
		});
		++g_jobs_active;
		g_jobs_peak = std::max(g_jobs_peak, g_jobs_active);
	}
	const auto admitted = std::chrono::steady_clock::now();
	const auto result = g_create_compute_pipelines(device, cache, count, infos, allocator, pipelines);
	const auto finished = std::chrono::steady_clock::now();
	const auto lookup = std::ranges::all_of(std::span(infos, count), [](const auto& info) {
		return (info.flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) != 0;
	});
	{
		std::lock_guard lock(g_jobs_mutex);
		--g_jobs_active;
		(lookup ? g_lookup_time : g_create_time).Record(result, admitted - asked, finished - admitted);
		if (result == VK_SUCCESS) {
			const auto previous = g_finished_pipelines;
			g_finished_pipelines += count;
			if (previous / 32 != g_finished_pipelines / 32) {
				std::fprintf(stderr, "kyty_precache: %llu compute pipelines ready, limit %u\n",
				             static_cast<unsigned long long>(g_finished_pipelines), g_jobs_limit);
			}
		}
	}
	g_jobs_available.notify_one();
	return result;
}
}

namespace Loader {
bool SystemContentParamSfoGetString(const char* name, std::string* value) {
	if (std::strcmp(name, "TITLE_ID") == 0) {
		*value = g_title;
		return true;
	}
	if (std::strcmp(name, "APP_VER") == 0) {
		*value = g_version;
		return true;
	}
	return false;
}
bool SystemContentParamSfoGetString(const char* name, char* value, size_t value_size) {
	std::string text;
	if (!SystemContentParamSfoGetString(name, &text) || text.size() + 1 > value_size) return false;
	std::memcpy(value, text.c_str(), text.size() + 1);
	return true;
}
}

unsigned int std::thread::hardware_concurrency() noexcept {
	const char* value = std::getenv("KYTY_PRECACHE_CPUS");
	const int   cpus  = value != nullptr ? std::atoi(value) : 6;
	return static_cast<unsigned>(std::clamp(cpus, 1, 16));
}

extern "C" void GuestThreadStackMappings_nid_no_patch(std::uintptr_t, std::uintptr_t, void (*)(void*, std::uintptr_t, std::uintptr_t),
                                                      void*) {}
extern "C" void GuestDeliveryHold_nid_no_patch() {}
extern "C" void GuestDeliveryRelease_nid_no_patch() {}
extern "C" void KernelPauseGuestClock_nid_no_patch() {}
extern "C" void KernelResumeGuestClock_nid_no_patch() {}

namespace Loader::GuestOodle {
bool Decompress(std::span<const uint8_t>, std::span<uint8_t>, const std::stop_token&) {
	return false;
}
}

namespace Libs::LibKernel::FileSystem {
std::filesystem::path GetRealFilename(const std::string&) {
	return {};
}
}

namespace Libs::Vr {
void OpenXrAppendExtensions(std::vector<const char*>&, const std::vector<vk::ExtensionProperties>&, bool) {}
}

namespace Libs::Graphics {
void ShowPrecompileNotice(const PrecompileNotice&) {}
void HidePrecompileNotice() {}
void ShowManifestProgress(const ManifestProgress&) {}
void HideManifestProgress() {}

struct PipelineCacheTestAccess {
	static PipelineCache::ManifestBacklog Backlog(const PipelineCache& cache) { return cache.GetManifestBacklog(); }
	static void Progress(PipelineCache& cache) { cache.ManifestProgress(true); }
};
}

namespace {
using namespace Libs::Graphics;

struct NoImageHeap final: ShaderRecompiler::IR::ImageHeapSource {
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest&, ShaderRecompiler::IR::ImageHeapView&) override { return false; }
};
int InventoryShaders(const char* assets, const char* destination, std::span<char*> input_images) {
	std::error_code error;
	const auto root = std::filesystem::canonical(assets, error);
	if (error || !std::filesystem::is_directory(root, error)) {
		std::fprintf(stderr, "kyty_precache: inventory requires a readable assets directory\n");
		return 66;
	}
	std::vector<std::filesystem::path> images;
	for (const auto* image: input_images) {
		auto path = std::filesystem::canonical(image, error);
		if (error || !std::filesystem::is_regular_file(path, error)) {
			std::fprintf(stderr, "kyty_precache: inventory image must be a readable file\n");
			return 66;
		}
		if (std::mismatch(root.begin(), root.end(), path.begin(), path.end()).first != root.end() &&
		    std::ranges::find(images, path) == images.end()) {
			images.push_back(std::move(path));
		}
	}
	const auto output = std::filesystem::weakly_canonical(destination, error);
	if (error || std::mismatch(root.begin(), root.end(), output.begin(), output.end()).first == root.end() ||
	    !std::filesystem::create_directory(output, error)) {
		std::fprintf(stderr, "kyty_precache: inventory requires a new output directory outside the assets directory\n");
		return 73;
	}
	const auto code_path = output / "shaders.code";
	const auto creation_path = output / "creation-records.bin";
	const auto shipped_path = output / "shipped.kpm";
	const auto report_path = output / "report.json";
	std::ofstream code_store(code_path, std::ios::binary);
	std::ofstream creation_records(creation_path, std::ios::binary);
	std::ofstream shipped_recipes(shipped_path, std::ios::binary);
	std::ofstream report(report_path);
	const bool written = code_store && creation_records && shipped_recipes && report &&
	                     WriteShaderInventory(root, images, code_store, creation_records, shipped_recipes, report);
	code_store.close();
	creation_records.close();
	shipped_recipes.close();
	report.close();
	if (written && code_store && creation_records && shipped_recipes && report) return 0;
	std::filesystem::remove(code_path, error);
	std::filesystem::remove(creation_path, error);
	std::filesystem::remove(shipped_path, error);
	std::filesystem::remove(report_path, error);
	std::filesystem::remove(output, error);
	std::fprintf(stderr, "kyty_precache: failed to write shader inventory\n");
	return 74;
}

}

int PrepareComputeRecipe(const char* signatures);

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "--inventory") == 0) {
		if (argc >= 4) return InventoryShaders(argv[2], argv[3], {argv + 4, static_cast<size_t>(argc - 4)});
		std::fprintf(stderr, "usage: kyty_precache --inventory <assets-directory> <new-output-directory> [image ...]\n");
		return 64;
	}
	if ((argc == 2 || argc == 3) && std::strcmp(argv[1], "--prepare-compute") == 0) {
		return PrepareComputeRecipe(argc == 3 ? argv[2] : nullptr);
	}
	const char* excluded_device = nullptr;
	if (argc >= 6 && std::strcmp(argv[argc - 2], "--exclude-device") == 0) {
		excluded_device = argv[argc - 1];
		argc -= 2;
	}
	if (argc < 4 || argc > 6 || (argc == 6 && std::strcmp(argv[5], "--modules-only") != 0)) {
		std::fprintf(stderr, "usage: kyty_precache <dossier contenant _PipelineCache> <TITLE_ID> <APP_VER> [secondes max=3600] [--modules-only]\n");
		return 64;
	}
	g_title   = argv[2];
	g_version = argv[3];
	const int limit = argc > 4 ? std::atoi(argv[4]) : 3600;
	if (chdir(argv[1]) != 0) {
		std::perror(argv[1]);
		return 2;
	}
	Common::InitializeThreads();
	KytyShim::StartRuntime();
	GraphicContext graphics;
	std::string    error;
	if (!CreateHeadlessVulkan(graphics, nullptr, error)) {
		std::fprintf(stderr, "kyty_precache: %s\n", error.c_str());
		return 3;
	}
	if (excluded_device) {
		const auto& device = graphics.GetPhysicalDeviceProperties();
		auto key = fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:", device.vendorID, device.deviceID, device.driverVersion);
		for (auto byte: device.pipelineCacheUUID) key += fmt::format("{:02x}", byte);
		if (key == excluded_device) {
			std::fprintf(stderr, "kyty_precache: batch contains quarantined compute programs for this device\n");
			return 77;
		}
	}
	KytyShim::ReportNativeRayTracing(graphics);
	g_jobs_limit = DriverComputeLimit();
	g_four_job_marker = std::getenv("KYTY_PRECACHE_FOUR_JOB_MARKER");
	g_create_compute_pipelines = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines = CreateComputePipelines;
	std::fprintf(stderr, "kyty_precache: GPU %s, store %s\n", graphics.physical_device_properties.deviceName.data(),
	             ShaderStorePath().string().c_str());
	std::fprintf(stderr, "kyty_precache: manifest workers=%u compute driver limit=%u\n",
	             std::max(1u, std::thread::hardware_concurrency() / 2), g_jobs_limit);
	graphics.image_heap_layout = ImageHeap::CreateLayout(graphics.device);
	NoImageHeap heap;
	bool completed = false;
	const auto  start = std::chrono::steady_clock::now();
	{
		PipelineCache cache(graphics, &heap, ShaderStorePath(), argc != 6, g_jobs_limit);
		std::fprintf(stderr, "kyty_precache: WarmUp done after %.1f s\n",
		             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
		uint32_t idle = 0;
		for (int second = 0; second < limit; second++) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			const auto backlog = PipelineCacheTestAccess::Backlog(cache);
			const bool empty   = backlog.modules == 0 && backlog.pipelines == 0 && backlog.running == 0 && !backlog.scanning;
			if (second % 5 == 0 || empty) {
				std::fprintf(stderr,
				             "kyty_precache: %d s, modules en file %u (dont calculs %u), pipelines en file %u, en cours %u, scan %d, "
				             "modules faits %llu, pipelines faits %llu\n",
				             second, backlog.modules, backlog.compute_modules, backlog.pipelines, backlog.running,
				             backlog.scanning ? 1 : 0, static_cast<unsigned long long>(backlog.modules_done),
				             static_cast<unsigned long long>(backlog.pipelines_done));
			}
			idle = empty ? idle + 1 : 0;
			if (idle >= 3) {
				completed = backlog.modules_total != 0 && backlog.modules_done == backlog.modules_total && backlog.failures == 0;
				break;
			}
		}
		PipelineCacheTestAccess::Progress(cache);
		cache.Save();
		std::fprintf(stderr, "kyty_precache: saved after %.1f s\n",
		             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
	}
	g_lookup_time.Report("lookup");
	g_create_time.Report("create");
	std::fprintf(stderr, "kyty_precache: compute_driver_peak=%u\n", g_jobs_peak);
	std::fprintf(stderr, "kyty_precache: done\n");
	std::fflush(stderr);
	std::_Exit(completed ? 0 : 124);
}
