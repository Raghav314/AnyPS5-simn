#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

struct BufferCacheBdaTestAccess {
	static uint64_t PageIndex(uint64_t address) { return BufferCache::PageIndex(address); }
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t Page        = BufferCache::CACHING_PAGESIZE;
constexpr uint64_t TableBlock  = 0x10000;
constexpr uint64_t Entry       = sizeof(vk::DeviceAddress);
constexpr uint64_t GuestBytes  = 3 * Page;
constexpr uint64_t FarBlocks   = 16;
constexpr int      CpuReadWrite = 0x03;

const char* g_name = "kyty_shim_bda_page_table";

std::vector<uint64_t> ReadTable(GraphicContext& graphics, CommandScheduler& scheduler, Buffer& table, uint64_t offset,
                                uint64_t size) {
	Buffer download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, size);
	download.CopyFrom(scheduler.Current(), table, offset, 0, size);
	scheduler.Finish();
	download.Invalidate(0, size);
	std::vector<uint64_t> entries(size / Entry);
	std::memcpy(entries.data(), download.Mapped().data(), size);
	return entries;
}

uint64_t BlockOffset(uint64_t page_index) {
	return page_index * Entry / TableBlock * TableBlock;
}

const char* CheckZero(const std::vector<uint64_t>& entries, const char* failure) {
	for (const auto entry: entries) {
		if (entry != 0) {
			return failure;
		}
	}
	return nullptr;
}

const char* CheckPageTable(GraphicContext& graphics, uint64_t address, bool sparse) {
	RenderContext   renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto& table = *buffers.GetBdaPageTableBuffer();

	const char* failure = nullptr;
	if (table.IsSparseResident() != sparse) {
		failure = sparse ? "the page table is not sparse-resident on a device with sparse residency buffers"
		                 : "the page table is sparse-resident without sparse residency buffers";
	}
	const auto first_page  = BufferCacheBdaTestAccess::PageIndex(address);
	const auto block       = BlockOffset(first_page);
	const auto far_block   = block + FarBlocks * TableBlock;
	if (failure == nullptr && far_block + TableBlock <= table.Size()) {
		failure = CheckZero(ReadTable(graphics, scheduler, table, far_block, TableBlock),
		                    "an entry of a page table block no buffer was registered in did not read 0");
	}
	if (failure == nullptr) {
		failure = CheckZero(ReadTable(graphics, scheduler, table, block, TableBlock),
		                    "an entry read before any buffer was registered in its block was not 0");
	}
	if (failure == nullptr) {
		(void)buffers.ObtainBuffer(address, GuestBytes, false, false);
		const auto id = buffers.RegisteredBuffer(address, Page);
		if (!id) {
			failure = "the buffer was not registered";
		} else {
			const auto& buffer  = buffers.GetBuffer(id);
			const auto  base    = BufferCacheBdaTestAccess::PageIndex(buffer.CpuAddress());
			const auto  pages   = (buffer.Size() + Page - 1) / Page;
			const auto  entries = ReadTable(graphics, scheduler, table, block, TableBlock);
			const auto  first   = block / Entry;
			for (uint64_t index = 0; index < entries.size() && failure == nullptr; index++) {
				const auto page     = first + index;
				const bool inside   = page >= base && page < base + pages;
				const auto expected = inside ? buffer.BufferDeviceAddress() + (page - base) * Page : 0;
				if (entries[index] != expected) {
					failure = inside ? "a registered page did not hold its buffer's device address"
					                 : "an entry next to a registered buffer in a newly backed block was not 0";
				}
			}
		}
	}
	(void)buffers.DropFreedRange(address, GuestBytes, nullptr, nullptr);
	if (failure == nullptr) {
		failure = CheckZero(ReadTable(graphics, scheduler, table, block, TableBlock),
		                    "an entry of a removed buffer did not read 0 again");
	}
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	return failure;
}

}

extern "C" int RunKytyBdaPageTableTest(int argc, char** argv) {
	const bool sparse = argc > 1 && std::string_view(argv[1]) == "sparse";
	g_name            = sparse ? "kyty_shim_bda_page_table_sparse" : "kyty_shim_bda_page_table_dense";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize(false, sparse)) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n", g_name, required ? "FAILED" : "skipped",
		             unavailable);
		return required ? 1 : 77;
	}
	if (sparse && !gpu.graphics.partial_buffers_enabled) {
		std::fprintf(stderr, "%s: skipped, the device has no strict sparse residency buffers\n", g_name);
		return 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", g_name);
		return 1;
	}
	void*              guest  = nullptr;
	constexpr uint64_t Mapped = GuestBytes + Page;
	if (sceKernelMapFlexibleMemory(&guest, Mapped, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", g_name);
		return 1;
	}
	const auto address = (reinterpret_cast<uint64_t>(guest) + Page - 1) / Page * Page;
	const auto* failure = CheckPageTable(gpu.graphics, address, sparse);
	if (sceKernelMunmap(reinterpret_cast<uint64_t>(guest), Mapped) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", g_name);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", g_name, failure);
		return 1;
	}
	std::printf("%s: ok\n", g_name);
	return 0;
}
