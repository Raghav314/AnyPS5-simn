#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_SURFACE_COPY_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_SURFACE_COPY_H_

#include <algorithm>
#include <cstdint>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

[[nodiscard]] inline vk::ImageCopy PresentationCopy(vk::Extent2D surface, vk::Extent3D backing,
                                                    uint32_t level, uint32_t layer) {
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, level, layer, 1};
	copy.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent = vk::Extent3D {std::min(surface.width, std::max(backing.width >> level, 1u)),
	                            std::min(surface.height, std::max(backing.height >> level, 1u)), 1};
	return copy;
}

}

#endif
