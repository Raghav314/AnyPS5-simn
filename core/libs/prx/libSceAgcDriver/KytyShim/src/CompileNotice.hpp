#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_COMPILENOTICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_COMPILENOTICE_HPP

#include "prx/libSceAgcDriver/Execution/include/KeyboardOverlay.hpp"

#include <vulkan/vulkan.h>

namespace KytyShim::CompileNotice {

std::uint64_t Revision() noexcept;
bool LayOut(AgcDriver::OverlayLayout& layout, VkExtent2D extent, bool loading);

}

#endif
