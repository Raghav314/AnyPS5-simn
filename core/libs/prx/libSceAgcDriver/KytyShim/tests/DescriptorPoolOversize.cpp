#include "TestVulkanDevice.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

bool RequireVulkan() {
	const char* value = std::getenv("ANYPS5_REQUIRE_VULKAN");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int SkipOrFail(const char* reason) {
	std::fprintf(stderr, "kyty_shim_descriptor_pool_oversize: %s\n", reason);
	return RequireVulkan() ? 1 : 77;
}

}

int main() {
	KytyShimTests::TestVulkanDevice vulkan;
	if (const char* reason = vulkan.Initialize(); reason != nullptr) {
		return SkipOrFail(reason);
	}

	const auto& limits = vulkan.graphics.GetPhysicalDeviceProperties().limits;
	constexpr uint32_t ExtraSampledImages = 16;
	constexpr uint32_t OversizeCount =
	    Libs::Graphics::DescriptorHeap::MaxSampledImages + ExtraSampledImages;
	if (limits.maxDescriptorSetSampledImages < OversizeCount ||
	    limits.maxPerStageDescriptorSampledImages < OversizeCount) {
		std::fprintf(stderr,
		             "kyty_shim_descriptor_pool_oversize: device sampled image limit (%u / %u) below %u\n",
		             limits.maxDescriptorSetSampledImages, limits.maxPerStageDescriptorSampledImages,
		             OversizeCount);
		return 77;
	}

	const std::array bindings {
	    vk::DescriptorSetLayoutBinding {
	        .binding         = 0,
	        .descriptorType  = vk::DescriptorType::eSampledImage,
	        .descriptorCount = OversizeCount,
	        .stageFlags      = vk::ShaderStageFlagBits::eCompute,
	    },
	    vk::DescriptorSetLayoutBinding {
	        .binding         = 1,
	        .descriptorType  = vk::DescriptorType::eStorageBuffer,
	        .descriptorCount = 1,
	        .stageFlags      = vk::ShaderStageFlagBits::eCompute,
	    },
	};

	const vk::DescriptorSetLayoutCreateInfo layout_info {
	    .bindingCount = static_cast<uint32_t>(bindings.size()),
	    .pBindings    = bindings.data(),
	};
	vk::DescriptorSetLayout layout;
	if (vulkan.graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &layout) !=
	    vk::Result::eSuccess) {
		std::fprintf(stderr,
		             "kyty_shim_descriptor_pool_oversize: FAILED createDescriptorSetLayout\n");
		return 1;
	}

	Libs::Graphics::DescriptorHeap::RegisterLayout(layout, bindings);

	int status = 0;
	{
		Libs::Graphics::MasterSemaphore master_semaphore(vulkan.graphics);
		Libs::Graphics::DescriptorHeap  heap(vulkan.graphics, master_semaphore);

		const auto regular_set = heap.Commit(layout);
		if (!regular_set) {
			std::fprintf(stderr,
			             "kyty_shim_descriptor_pool_oversize: FAILED Commit returned null set\n");
			status = 1;
		}

		const auto persistent_set = heap.CommitPersistent(layout);
		if (!persistent_set) {
			std::fprintf(stderr,
			             "kyty_shim_descriptor_pool_oversize: FAILED CommitPersistent returned null set\n");
			status = 1;
		}

		const vk::SemaphoreSignalInfo signal_info {
		    .semaphore = master_semaphore.Handle(),
		    .value     = master_semaphore.CurrentTick(),
		};
		if (vulkan.graphics.device.signalSemaphore(&signal_info) != vk::Result::eSuccess) {
			std::fprintf(stderr,
			             "kyty_shim_descriptor_pool_oversize: FAILED signalSemaphore\n");
			status = 1;
		}
	}

	vulkan.graphics.device.destroyDescriptorSetLayout(layout, nullptr);
	if (status == 0) {
		std::printf("kyty_shim_descriptor_pool_oversize: ok\n");
	}
	return status;
}
