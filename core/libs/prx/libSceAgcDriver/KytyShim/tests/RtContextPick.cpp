#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterRayQueryShared.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace Emit       = Recompiler::Spirv::Emitter;
namespace Rq         = Recompiler::Spirv::Emitter::RayQueryEmit;

constexpr const char* Name        = "kyty_shim_rt_context_pick";
constexpr int         SkippedExit = 77;
constexpr uint32_t    Lanes       = 64;
constexpr uint32_t    LaneWords   = 4;
constexpr uint32_t    LaneBase    = 4;
constexpr uint32_t    Entries     = 4;
constexpr uint32_t    Miss        = 0xffffffffu;
constexpr uint32_t    NoHit       = 0xfffffffeu;
constexpr uint64_t    HeaderA     = 0x10001000ull;
constexpr uint64_t    HeaderB     = 0x10002000ull;
constexpr uint64_t    HeaderStale = 0x10003000ull;
constexpr uint64_t    HeaderNone  = 0x10004000ull;
constexpr uint32_t    IndexA      = 11;
constexpr uint32_t    IndexB      = 22;

uint32_t g_failures = 0;

void Require(bool value, const std::string& what) {
	if (value) return;
	std::fprintf(stderr, "%s: %s\n", Name, what.c_str());
	std::exit(1);
}

void Check(bool value, const std::string& what) {
	if (value) return;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
	++g_failures;
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result.result));
	return result.value;
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

struct Gpu {
	vk::Instance                       instance;
	vk::PhysicalDevice                 physical;
	vk::Device                         device;
	vk::Queue                          queue;
	uint32_t                           family = 0;
	vk::PhysicalDeviceMemoryProperties memory;
	uint32_t                           scratch_alignment = 256;
};

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0, std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s: skipped, %s\n", Name, why);
	return SkippedExit;
}

std::optional<Gpu> OpenGpu(const char*& why) {
	static const vk::detail::DynamicLoader loader;
	if (!loader.success()) {
		why = "the Vulkan loader is missing";
		return std::nullopt;
	}
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) {
		why = "no Vulkan 1.3 instance";
		return std::nullopt;
	}
	Gpu gpu;
	gpu.instance = created.value;
	dispatch.init(gpu.instance);
	const std::array<const char*, 3> extensions {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
	                                             VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
	for (const auto candidate: Checked(gpu.instance.enumeratePhysicalDevices(), "device enumeration")) {
		if (candidate.getProperties().apiVersion < VK_API_VERSION_1_3) continue;
		const auto available = Checked(candidate.enumerateDeviceExtensionProperties(), "extension enumeration");
		const bool has_all   = std::ranges::all_of(extensions, [&](const char* name) {
            return std::ranges::any_of(available, [&](const auto& each) { return std::strcmp(each.extensionName, name) == 0; });
        });
		if (!has_all) continue;
		const auto families = candidate.getQueueFamilyProperties();
		const auto compute  = std::ranges::find_if(families, [](const auto& queue) {
            return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);
        });
		if (compute == families.end()) continue;
		vk::PhysicalDeviceRayQueryFeaturesKHR              ray_query;
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR structures;
		vk::PhysicalDeviceVulkan12Features                 features12;
		vk::PhysicalDeviceFeatures2                        features;
		structures.pNext = &ray_query;
		features12.pNext = &structures;
		features.pNext   = &features12;
		candidate.getFeatures2(&features);
		if (!ray_query.rayQuery || !structures.accelerationStructure || !features12.bufferDeviceAddress ||
		    !features.features.shaderInt64) {
			continue;
		}
		gpu.physical = candidate;
		gpu.family   = static_cast<uint32_t>(compute - families.begin());
		break;
	}
	if (!gpu.physical) {
		gpu.instance.destroy();
		why = "no Vulkan 1.3 device with ray queries, acceleration structures, 64-bit integers and buffer device addresses";
		return std::nullopt;
	}
	vk::PhysicalDeviceRayQueryFeaturesKHR              enable_query;
	vk::PhysicalDeviceAccelerationStructureFeaturesKHR enable_structures;
	vk::PhysicalDeviceVulkan12Features                 enable12;
	vk::PhysicalDeviceFeatures2                        enable;
	enable_query.rayQuery                  = VK_TRUE;
	enable_structures.accelerationStructure = VK_TRUE;
	enable_structures.pNext                = &enable_query;
	enable12.bufferDeviceAddress           = VK_TRUE;
	enable12.pNext                         = &enable_structures;
	enable.features.shaderInt64            = VK_TRUE;
	enable.pNext                           = &enable12;
	const float               priority     = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = gpu.family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	vk::DeviceCreateInfo device_info;
	device_info.pNext                   = &enable;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	gpu.device                          = Checked(gpu.physical.createDevice(device_info), "device creation");
	dispatch.init(gpu.device);
	gpu.queue  = gpu.device.getQueue(gpu.family, 0);
	gpu.memory = gpu.physical.getMemoryProperties();
	vk::PhysicalDeviceAccelerationStructurePropertiesKHR structure_properties;
	vk::PhysicalDeviceProperties2                        properties;
	properties.pNext = &structure_properties;
	gpu.physical.getProperties2(&properties);
	gpu.scratch_alignment = std::max(structure_properties.minAccelerationStructureScratchOffsetAlignment, 1u);
	return gpu;
}

struct HostBuffer {
	vk::Buffer        buffer;
	vk::DeviceMemory  memory;
	void*             mapped  = nullptr;
	uint64_t          bytes   = 0;
	vk::DeviceAddress address = 0;
};

HostBuffer CreateBuffer(const Gpu& gpu, uint64_t bytes, vk::BufferUsageFlags usage) {
	HostBuffer           result;
	vk::BufferCreateInfo info;
	result.bytes  = std::max<uint64_t>((bytes + 255u) & ~uint64_t {255u}, 256u);
	info.size     = result.bytes;
	info.usage    = usage | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	result.buffer = Checked(gpu.device.createBuffer(info), "buffer creation");
	const auto needs = gpu.device.getBufferMemoryRequirements(result.buffer);
	const auto flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	for (uint32_t type = 0; type < gpu.memory.memoryTypeCount && !result.memory; type++) {
		if ((needs.memoryTypeBits >> type & 1u) == 0u || (gpu.memory.memoryTypes[type].propertyFlags & flags) != flags) continue;
		vk::MemoryAllocateFlagsInfo allocate_flags;
		allocate_flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
		vk::MemoryAllocateInfo allocate;
		allocate.pNext           = &allocate_flags;
		allocate.allocationSize  = needs.size;
		allocate.memoryTypeIndex = type;
		result.memory            = Checked(gpu.device.allocateMemory(allocate), "memory allocation");
	}
	Require(static_cast<bool>(result.memory), "the device has no host-visible memory for a buffer");
	Checked(gpu.device.bindBufferMemory(result.buffer, result.memory, 0), "buffer binding");
	result.mapped = Checked(gpu.device.mapMemory(result.memory, 0, VK_WHOLE_SIZE), "buffer mapping");
	std::memset(result.mapped, 0, result.bytes);
	vk::BufferDeviceAddressInfo address_info;
	address_info.buffer = result.buffer;
	result.address      = gpu.device.getBufferAddress(address_info);
	return result;
}

void DestroyBuffer(const Gpu& gpu, HostBuffer& buffer) {
	gpu.device.unmapMemory(buffer.memory);
	gpu.device.destroyBuffer(buffer.buffer);
	gpu.device.freeMemory(buffer.memory);
}

struct Structure {
	HostBuffer                   storage;
	HostBuffer                   scratch;
	vk::AccelerationStructureKHR handle;
	vk::DeviceAddress            address = 0;
};

constexpr auto BuildInput = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

void Build(const Gpu& gpu, vk::CommandBuffer command, Structure& structure, vk::AccelerationStructureTypeKHR type,
           const vk::AccelerationStructureGeometryKHR& geometry) {
	vk::AccelerationStructureBuildGeometryInfoKHR info;
	info.type          = type;
	info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = 1;
	info.pGeometries   = &geometry;
	const uint32_t primitives = 1;
	vk::AccelerationStructureBuildSizesInfoKHR sizes;
	gpu.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info, &primitives, &sizes);
	structure.storage = CreateBuffer(gpu, sizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR);
	structure.scratch = CreateBuffer(gpu, sizes.buildScratchSize + gpu.scratch_alignment, vk::BufferUsageFlagBits::eStorageBuffer);
	vk::AccelerationStructureCreateInfoKHR create;
	create.buffer    = structure.storage.buffer;
	create.size      = sizes.accelerationStructureSize;
	create.type      = type;
	structure.handle = Checked(gpu.device.createAccelerationStructureKHR(create), "acceleration structure creation");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info;
	address_info.accelerationStructure = structure.handle;
	structure.address                  = gpu.device.getAccelerationStructureAddressKHR(address_info);
	info.dstAccelerationStructure      = structure.handle;
	info.scratchData.deviceAddress =
	    (structure.scratch.address + gpu.scratch_alignment - 1u) & ~vk::DeviceAddress {gpu.scratch_alignment - 1u};
	vk::AccelerationStructureBuildRangeInfoKHR range;
	range.primitiveCount = 1;
	const auto* ranges   = &range;
	command.buildAccelerationStructuresKHR(1, &info, &ranges);
	vk::MemoryBarrier barrier;
	barrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
	barrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
	                        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eComputeShader, {},
	                        1, &barrier, 0, nullptr, 0, nullptr);
}

void DestroyStructure(const Gpu& gpu, Structure& structure) {
	gpu.device.destroyAccelerationStructureKHR(structure.handle);
	DestroyBuffer(gpu, structure.storage);
	DestroyBuffer(gpu, structure.scratch);
}

std::vector<uint32_t> BuildModule() {
	Recompiler::IR::Program program;
	program.stage = ShaderType::Compute;
	const ShaderComputeInputInfo compute {};
	Recompiler::Spirv::Profile   profile;
	Emit::EmitterState           state(program, {.compute = &compute}, profile);
	auto&                        builder = state.builder;
	builder.RequireVersion(0x00010600u);
	builder.RequireCapability(spv::CapabilityShader);
	builder.RequireCapability(spv::CapabilityInt64);
	builder.RequireCapability(spv::CapabilityRayQueryKHR);
	builder.RequireCapability(spv::CapabilityPhysicalStorageBufferAddresses);
	builder.RequireExtension("SPV_KHR_ray_query");
	builder.AddMemoryModel(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);
	const auto u32           = Emit::TypeU32(state);
	const auto u64           = Emit::TypeU64(state);
	const auto boolean       = Emit::TypeBool(state);
	const auto f32           = Emit::TypeF32(state);
	const auto vec3          = Emit::TypeF32Vector(state, 3);
	const auto function      = builder.AllocateId();
	const auto global_type   = Emit::TypeU32Vector(state, 3);
	const auto input_pointer = builder.Type(spv::OpTypePointer, spv::StorageClassInput, global_type);
	const auto global_id     = builder.DefineGlobalVariable(input_pointer, spv::StorageClassInput);
	builder.AddAnnotation(spv::OpDecorate, global_id, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
	const auto array_type = builder.Type(spv::OpTypeRuntimeArray, u32);
	builder.AddAnnotation(spv::OpDecorate, array_type, spv::DecorationArrayStride, 4u);
	const auto buffer_type = builder.Type(spv::OpTypeStruct, array_type);
	builder.AddAnnotation(spv::OpDecorate, buffer_type, spv::DecorationBlock);
	builder.AddAnnotation(spv::OpMemberDecorate, buffer_type, 0u, spv::DecorationOffset, 0u);
	const auto buffer_pointer = builder.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, buffer_type);
	const auto buffer         = builder.DefineGlobalVariable(buffer_pointer, spv::StorageClassStorageBuffer);
	builder.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationDescriptorSet, 0u);
	builder.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationBinding, 0u);
	const auto word_pointer = builder.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, u32);
	const auto query_type   = builder.Type(spv::OpTypeRayQueryKHR);
	const auto query        = builder.DefineGlobalVariable(builder.Type(spv::OpTypePointer, spv::StorageClassPrivate, query_type),
	                                                       spv::StorageClassPrivate);
	builder.AddEntryPoint(spv::ExecutionModelGLCompute, function, "main", {global_id});
	builder.AddExecutionMode(function, spv::ExecutionModeLocalSize, Lanes, 1u, 1u);
	builder.AddFunction(spv::OpFunction, Emit::TypeVoid(state), function, spv::FunctionControlMaskNone, Emit::TypeFunction(state));
	Emit::EmitLabel(state, builder.AllocateId());
	const auto invocation = Emit::Unary(state, spv::OpLoad, global_type, global_id);
	const auto lane       = builder.AllocateId();
	builder.AddFunction(spv::OpCompositeExtract, u32, lane, invocation, 0u);
	const auto word = [&](uint32_t position) {
		const auto pointer = builder.AllocateId();
		builder.AddFunction(spv::OpAccessChain, word_pointer, pointer, buffer, Emit::ConstantU32(state, 0), position);
		return pointer;
	};
	const auto lane_word = [&](uint32_t offset) {
		const auto first = Emit::Binary(state, spv::OpIMul, u32, lane, Emit::ConstantU32(state, LaneWords));
		return word(Emit::Binary(state, spv::OpIAdd, u32, first, Emit::ConstantU32(state, LaneBase + offset)));
	};
	const auto context = Emit::PackU64(state, Emit::Unary(state, spv::OpLoad, u32, word(Emit::ConstantU32(state, 0))),
	                                   Emit::Unary(state, spv::OpLoad, u32, word(Emit::ConstantU32(state, 1))));
	const auto header =
	    Emit::PackU64(state, Emit::Unary(state, spv::OpLoad, u32, lane_word(0)), Emit::Unary(state, spv::OpLoad, u32, lane_word(1)));
	const auto chosen = Rq::PickContext(state, Emit::ConstantBool(state, true), context, header, Entries);
	const auto found  = Emit::Binary(state, spv::OpINotEqual, boolean, chosen, Emit::ConstantU64(state, 0));
	const auto result = Emit::EmitValueOrDefaultIfCondition(state, found, u32, Emit::ConstantU32(state, Miss), [&] {
		const auto tlas      = Rq::LoadAddressU64(state, Rq::AddressPlus(state, chosen, Rq::ContextTlas));
		const auto structure = Emit::Unary(state, spv::OpConvertUToAccelerationStructureKHR,
		                                   builder.Type(spv::OpTypeAccelerationStructureKHR), tlas);
		const auto zero      = Emit::ConstantF32Value(state, 0.0f);
		const auto origin    = builder.AllocateId();
		builder.AddFunction(spv::OpCompositeConstruct, vec3, origin, zero, zero, zero);
		const auto direction = builder.AllocateId();
		builder.AddFunction(spv::OpCompositeConstruct, vec3, direction, zero, zero, Emit::ConstantF32Value(state, 1.0f));
		builder.AddFunction(spv::OpRayQueryInitializeKHR, query, structure, Emit::ConstantU32(state, spv::RayFlagsOpaqueKHRMask),
		                    Emit::ConstantU32(state, 0xffu), origin, Emit::ConstantF32Value(state, 0.001f), direction,
		                    Emit::ConstantF32Value(state, 100.0f));
		const auto proceed = builder.AllocateId();
		builder.AddFunction(spv::OpRayQueryProceedKHR, boolean, proceed, query);
		const auto type = builder.AllocateId();
		builder.AddFunction(spv::OpRayQueryGetIntersectionTypeKHR, u32, type, query, Emit::ConstantU32(state, 1));
		const auto custom = builder.AllocateId();
		builder.AddFunction(spv::OpRayQueryGetIntersectionInstanceCustomIndexKHR, Emit::TypeI32(state), custom, query,
		                    Emit::ConstantU32(state, 1));
		const auto hit = Emit::Binary(state, spv::OpINotEqual, boolean, type, Emit::ConstantU32(state, 0));
		return Emit::Select(state, u32, hit, Emit::Unary(state, spv::OpBitcast, u32, custom), Emit::ConstantU32(state, NoHit));
	});
	builder.AddFunction(spv::OpStore, lane_word(2), result);
	(void)f32;
	builder.AddFunction(spv::OpReturn);
	builder.AddFunction(spv::OpFunctionEnd);
	auto                 words = builder.Build();
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	Require(tools.Validate(words), "the context pick program is not valid SPIR-V: " + messages);
	return words;
}

void WriteEntry(uint8_t* table, uint32_t index, uint64_t header, uint64_t tlas, uint32_t valid) {
	auto* entry = table + uint64_t {index} * Rq::ContextStride;
	std::memcpy(entry + Rq::ContextHeader, &header, sizeof(header));
	std::memcpy(entry + Rq::ContextTlas, &tlas, sizeof(tlas));
	std::memcpy(entry + Rq::ContextValid, &valid, sizeof(valid));
}

vk::AccelerationStructureInstanceKHR Instance(uint64_t blas, float depth, uint32_t custom) {
	vk::AccelerationStructureInstanceKHR instance;
	instance.transform.matrix[0][0]               = 1.0f;
	instance.transform.matrix[1][1]               = 1.0f;
	instance.transform.matrix[2][2]               = 1.0f;
	instance.transform.matrix[2][3]               = depth;
	instance.instanceCustomIndex                  = custom;
	instance.mask                                 = 0xffu;
	instance.flags                                = static_cast<VkGeometryInstanceFlagsKHR>(
        vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable | vk::GeometryInstanceFlagBitsKHR::eForceOpaque);
	instance.accelerationStructureReference       = blas;
	return instance;
}

void Run(const Gpu& gpu) {
	const auto words = BuildModule();

	constexpr std::array<float, 9> vertices {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
	auto vertex_buffer = CreateBuffer(gpu, sizeof(vertices), BuildInput);
	std::memcpy(vertex_buffer.mapped, vertices.data(), sizeof(vertices));
	vk::AccelerationStructureGeometryKHR triangles;
	triangles.geometryType                                = vk::GeometryTypeKHR::eTriangles;
	triangles.flags                                       = vk::GeometryFlagBitsKHR::eOpaque;
	triangles.geometry.triangles                          = vk::AccelerationStructureGeometryTrianglesDataKHR {};
	triangles.geometry.triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat;
	triangles.geometry.triangles.vertexData.deviceAddress = vertex_buffer.address;
	triangles.geometry.triangles.vertexStride             = sizeof(float) * 3;
	triangles.geometry.triangles.maxVertex                = 2;
	triangles.geometry.triangles.indexType                = vk::IndexType::eNoneKHR;

	vk::CommandPoolCreateInfo pool_info;
	pool_info.queueFamilyIndex = gpu.family;
	const auto command_pool    = Checked(gpu.device.createCommandPool(pool_info), "command pool creation");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = command_pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command;
	Checked(gpu.device.allocateCommandBuffers(&command_info, &command), "command buffer allocation");
	vk::CommandBufferBeginInfo begin;
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	Checked(command.begin(begin), "command buffer begin");

	Structure blas;
	Build(gpu, command, blas, vk::AccelerationStructureTypeKHR::eBottomLevel, triangles);
	std::array<Structure, 2>  tlases;
	std::array<HostBuffer, 2> instance_buffers;
	const std::array<std::pair<float, uint32_t>, 2> placed {{{1.0f, IndexA}, {2.0f, IndexB}}};
	for (size_t at = 0; at < tlases.size(); at++) {
		instance_buffers[at] = CreateBuffer(gpu, sizeof(vk::AccelerationStructureInstanceKHR), BuildInput);
		const auto instance  = Instance(blas.address, placed[at].first, placed[at].second);
		std::memcpy(instance_buffers[at].mapped, &instance, sizeof(instance));
		vk::AccelerationStructureGeometryKHR instances;
		instances.geometryType                       = vk::GeometryTypeKHR::eInstances;
		instances.geometry.instances                 = vk::AccelerationStructureGeometryInstancesDataKHR {};
		instances.geometry.instances.arrayOfPointers = VK_FALSE;
		instances.geometry.instances.data.deviceAddress = instance_buffers[at].address;
		Build(gpu, command, tlases[at], vk::AccelerationStructureTypeKHR::eTopLevel, instances);
	}

	auto  table = CreateBuffer(gpu, uint64_t {Entries} * Rq::ContextStride, vk::BufferUsageFlagBits::eStorageBuffer);
	auto* bytes = static_cast<uint8_t*>(table.mapped);
	WriteEntry(bytes, 0, HeaderA, tlases[0].address, 1);
	WriteEntry(bytes, 1, HeaderB, tlases[1].address, 1);
	WriteEntry(bytes, 2, HeaderStale, tlases[0].address, 0);
	WriteEntry(bytes, 3, HeaderA, tlases[1].address, 1);

	auto  data   = CreateBuffer(gpu, (LaneBase + Lanes * LaneWords) * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
	auto* values = static_cast<uint32_t*>(data.mapped);
	values[0]    = static_cast<uint32_t>(table.address);
	values[1]    = static_cast<uint32_t>(table.address >> 32u);
	const std::array<uint64_t, 4> headers {HeaderA, HeaderB, HeaderStale, HeaderNone};
	for (uint32_t lane = 0; lane < Lanes; lane++) {
		const auto header                           = headers[lane % headers.size()];
		values[LaneBase + lane * LaneWords + 0]     = static_cast<uint32_t>(header);
		values[LaneBase + lane * LaneWords + 1]     = static_cast<uint32_t>(header >> 32u);
		values[LaneBase + lane * LaneWords + 2]     = 0x12345678u;
	}

	const vk::DescriptorSetLayoutBinding binding {0u, vk::DescriptorType::eStorageBuffer, 1u, vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo set_info;
	set_info.bindingCount = 1;
	set_info.pBindings    = &binding;
	const auto set_layout = Checked(gpu.device.createDescriptorSetLayout(set_info), "set layout creation");
	vk::PipelineLayoutCreateInfo layout_info;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &set_layout;
	const auto layout          = Checked(gpu.device.createPipelineLayout(layout_info), "pipeline layout creation");
	vk::ShaderModuleCreateInfo module_info;
	module_info.codeSize = words.size() * sizeof(uint32_t);
	module_info.pCode    = words.data();
	const auto module    = Checked(gpu.device.createShaderModule(module_info), "module creation");
	vk::ComputePipelineCreateInfo pipeline_info;
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = layout;
	vk::Pipeline pipeline;
	Checked(gpu.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline), "pipeline creation");
	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eStorageBuffer, 1u};
	vk::DescriptorPoolCreateInfo descriptor_pool_info;
	descriptor_pool_info.maxSets       = 1;
	descriptor_pool_info.poolSizeCount = 1;
	descriptor_pool_info.pPoolSizes    = &pool_size;
	const auto descriptor_pool         = Checked(gpu.device.createDescriptorPool(descriptor_pool_info), "descriptor pool creation");
	vk::DescriptorSetAllocateInfo allocate;
	allocate.descriptorPool     = descriptor_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &set_layout;
	vk::DescriptorSet set;
	Checked(gpu.device.allocateDescriptorSets(&allocate, &set), "descriptor set allocation");
	const vk::DescriptorBufferInfo buffer_info {data.buffer, 0u, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet write;
	write.dstSet          = set;
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &buffer_info;
	gpu.device.updateDescriptorSets(1, &write, 0, nullptr);

	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &set, 0, nullptr);
	command.dispatch(1, 1, 1);
	vk::MemoryBarrier barrier;
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0, nullptr, 0,
	                        nullptr);
	Checked(command.end(), "command buffer end");
	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Checked(gpu.queue.submit(1, &submit, nullptr), "submission");
	Checked(gpu.queue.waitIdle(), "queue wait");

	const std::array<uint32_t, 4> expected {IndexA, IndexB, Miss, Miss};
	const std::array<const char*, 4> what {"a walk on tree A hits tree A, not the later entry with the same header",
	                                       "a walk on tree B hits tree B", "a walk never picks an entry that is not valid",
	                                       "a walk whose tree has no entry picks none"};
	for (uint32_t lane = 0; lane < Lanes; lane++) {
		const auto got = values[LaneBase + lane * LaneWords + 2];
		const auto at  = lane % expected.size();
		Check(got == expected[at], std::string(what[at]) + " (lane " + std::to_string(lane) + ": " + std::to_string(got) + ")");
	}

	gpu.device.destroyPipeline(pipeline);
	gpu.device.destroyShaderModule(module);
	gpu.device.destroyPipelineLayout(layout);
	gpu.device.destroyDescriptorPool(descriptor_pool);
	gpu.device.destroyDescriptorSetLayout(set_layout);
	gpu.device.destroyCommandPool(command_pool);
	DestroyBuffer(gpu, data);
	DestroyBuffer(gpu, table);
	for (auto& each: tlases) DestroyStructure(gpu, each);
	for (auto& each: instance_buffers) DestroyBuffer(gpu, each);
	DestroyStructure(gpu, blas);
	DestroyBuffer(gpu, vertex_buffer);
}

}

int main() {
	const char* why = nullptr;
	auto        gpu = OpenGpu(why);
	if (!gpu) return NoDevice(why);
	Run(*gpu);
	gpu->device.destroy();
	gpu->instance.destroy();
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u failures\n", Name, g_failures);
		return 1;
	}
	std::printf("%s: ok\n", Name);
	return 0;
}
