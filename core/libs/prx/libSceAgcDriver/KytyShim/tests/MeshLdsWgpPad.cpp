#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name = "kyty_shim_mesh_lds_wgp_pad";

constexpr std::array<uint32_t, 28> MeshWithLdsCode {
    0xbefc03ffu, 0x00001003u, 0xbf900009u, 0x7e0202f3u, 0x7e0402f3u, 0x7e060280u, 0x7e0802f2u,
    0x7e0a02ffu, 0x40400000u, 0x7d840084u, 0x02020b01u, 0x7d840088u, 0x02040b02u, 0x340c0082u,
    0xd8340000u, 0x00000406u, 0xbf8a0000u, 0xd8d80000u, 0x07000006u, 0xbf8cc07fu, 0x7e1002ffu,
    0x00200400u, 0xf80000cfu, 0x04030201u, 0xf800020fu, 0x04040307u, 0xf8000941u, 0x00000008u,
};

constexpr std::array<uint32_t, 72> InterpPixelSpirv {
    0x07230203u, 0x00010400u, 0x00070000u, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
    0x00000003u, 0x00030010u, 0x00000001u, 0x00000007u, 0x00040047u, 0x00000002u, 0x0000001eu, 0x00000000u,
    0x00040047u, 0x00000003u, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000004u, 0x00030021u, 0x00000005u,
    0x00000004u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u,
    0x00040020u, 0x00000008u, 0x00000001u, 0x00000007u, 0x00040020u, 0x00000009u, 0x00000003u, 0x00000007u,
    0x0004003bu, 0x00000008u, 0x00000002u, 0x00000001u, 0x0004003bu, 0x00000009u, 0x00000003u, 0x00000003u,
    0x00050036u, 0x00000004u, 0x00000001u, 0x00000000u, 0x00000005u, 0x000200f8u, 0x0000000au, 0x0004003du,
    0x00000007u, 0x0000000bu, 0x00000002u, 0x0003003eu, 0x00000003u, 0x0000000bu, 0x000100fdu, 0x00010038u,
};

constexpr std::array<uint32_t, 129> AliasedWorkgroupSpirv {
    0x07230203u, 0x00010400u, 0x00070000u, 0x00000011u, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x0000000bu, 0x00020011u, 0x0000114cu, 0x000c000au, 0x5f565053u, 0x5f52484bu, 0x6b726f77u, 0x756f7267u,
    0x656d5f70u, 0x79726f6du, 0x7078655fu, 0x6963696cu, 0x616c5f74u, 0x74756f79u, 0x00000000u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0007000fu, 0x00000005u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
    0x00000003u, 0x00060010u, 0x00000001u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00040047u,
    0x00000004u, 0x00000006u, 0x00000004u, 0x00040047u, 0x00000005u, 0x00000006u, 0x00000008u, 0x00050048u,
    0x00000006u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x00000006u, 0x00000002u, 0x00050048u,
    0x00000007u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x00000007u, 0x00000002u, 0x00030047u,
    0x00000002u, 0x00000014u, 0x00030047u, 0x00000003u, 0x00000014u, 0x00020013u, 0x00000008u, 0x00030021u,
    0x00000009u, 0x00000008u, 0x00040015u, 0x0000000au, 0x00000020u, 0x00000000u, 0x00040015u, 0x0000000bu,
    0x00000040u, 0x00000000u, 0x0004002bu, 0x0000000au, 0x0000000cu, 0x00000010u, 0x0004002bu, 0x0000000au,
    0x0000000du, 0x00000008u, 0x0004001cu, 0x00000004u, 0x0000000au, 0x0000000cu, 0x0004001cu, 0x00000005u,
    0x0000000bu, 0x0000000du, 0x0003001eu, 0x00000006u, 0x00000004u, 0x0003001eu, 0x00000007u, 0x00000005u,
    0x00040020u, 0x0000000eu, 0x00000004u, 0x00000006u, 0x00040020u, 0x0000000fu, 0x00000004u, 0x00000007u,
    0x0004003bu, 0x0000000eu, 0x00000002u, 0x00000004u, 0x0004003bu, 0x0000000fu, 0x00000003u, 0x00000004u,
    0x00050036u, 0x00000008u, 0x00000001u, 0x00000000u, 0x00000009u, 0x000200f8u, 0x00000010u, 0x000100fdu,
    0x00010038u,
};

void Require(bool condition, const std::string& message) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, message.c_str());
	std::exit(1);
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess,
	        std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0,
	        std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s_draw: skipped, %s\n", Name, why);
	return 77;
}

Recompiler::CompileResult CompileMeshWithLds(uint32_t lds_dwords, uint32_t host_subgroup_size, bool pad) {
	std::vector<uint32_t> code(MeshWithLdsCode.begin(), MeshWithLdsCode.end());
	code.push_back(0xbf810000u);

	ShaderVertexInputInfo vertex {};
	vertex.logical_stage              = ShaderType::Mesh;
	vertex.wave_size                  = 64u;
	vertex.mesh.lds_size_dwords       = lds_dwords;
	vertex.mesh.threads_num[0]        = 64u;
	vertex.mesh.threads_num[1]        = 1u;
	vertex.mesh.threads_num[2]        = 1u;
	vertex.mesh.host_subgroup_size    = host_subgroup_size;
	vertex.mesh.vertices_per_group    = 3u;
	vertex.mesh.primitives_per_group  = 1u;
	vertex.mesh.max_vertices          = 64u;
	vertex.mesh.max_primitives        = 1u;
	vertex.mesh.input_primitive       = static_cast<uint32_t>(Prospero::PrimitiveType::kPointList);
	vertex.mesh.output_primitive_size = 3u;
	vertex.mesh.provoking_vertex      = 0u;
	vertex.mesh.fast_launch           = false;

	std::array<uint32_t, 2>    user_data {};
	Recompiler::Spirv::Profile profile {.mesh_lds_padding = pad};
	Recompiler::CompileOptions options {};
	options.stage             = ShaderType::Mesh;
	options.wave_size         = 64u;
	options.user_data_base    = 8u;
	options.user_data         = user_data;
	options.shader_hash       = 0x6d6573686c647300ull ^ lds_dwords;
	options.profile           = &profile;
	options.input_info.vertex = &vertex;

	const auto                       source     = Recompiler::PrepareShaderSource(code, options);
	auto                             translated = Recompiler::TranslateProgram(source.decoded, options);
	const auto                       plan       = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime runtime {};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	Require(Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
	        "failed to materialize mesh resources");
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization,
	                                           Recompiler::IR::PushData::MeshDrawDwordCount);

	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer(
	    [&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		    messages += message;
		    messages += '\n';
	    });
	spvtools::ValidatorOptions validator;
	validator.SetUniformBufferStandardLayout(true);
	validator.SetScalarBlockLayout(true);
	const bool valid = tools.Validate(compiled.spirv.data(), compiled.spirv.size(), validator);
	Require(valid, std::string("SPIR-V validation failed:\n") + messages);
	return compiled;
}

void VerifyCodegenAndSizing() {
	static_assert(NeedsMeshLdsPadding(0x1002, 0x73bf, vk::DriverId::eAmdProprietary));
	static_assert(!NeedsMeshLdsPadding(0x10de, 0x2702, vk::DriverId::eNvidiaProprietary));
	static_assert(!NeedsMeshLdsPadding(0x10de, 0x73bf, vk::DriverId::eAmdProprietary));
	static_assert(!NeedsMeshLdsPadding(0x1002, 0x73bf, vk::DriverId::eMesaRadv));
	static_assert(!NeedsMeshLdsPadding(0x1002, 0x744c, vk::DriverId::eAmdProprietary));
	for (const uint32_t subgroup: {32u, 64u}) {
		constexpr uint32_t LdsDwords = 3200u;
		const auto         compiled  = CompileMeshWithLds(LdsDwords, subgroup, true);
		const auto&        words     = compiled.spirv;

		std::map<uint32_t, uint32_t>               spec_ids;
		std::map<uint32_t, uint32_t>               spec_defaults;
		std::map<uint32_t, std::pair<uint32_t, uint32_t>> array_types;
		std::map<uint32_t, uint32_t>               wg_ptrs;
		std::set<uint32_t>                         wg_var_types;
		for (size_t at = 5; at < words.size();) {
			const uint32_t count = words[at] >> spv::WordCountShift;
			if (count == 0 || at + count > words.size()) break;
			const auto op = static_cast<spv::Op>(words[at] & spv::OpCodeMask);
			if (op == spv::OpDecorate && count >= 4u &&
			    static_cast<spv::Decoration>(words[at + 2u]) == spv::DecorationSpecId) {
				spec_ids[words[at + 1u]] = words[at + 3u];
			} else if (op == spv::OpSpecConstant && count == 4u) {
				spec_defaults[words[at + 2u]] = words[at + 3u];
			} else if (op == spv::OpTypeArray && count >= 4u) {
				array_types[words[at + 1u]] = {words[at + 2u], words[at + 3u]};
			} else if (op == spv::OpTypePointer && count >= 4u &&
			           static_cast<spv::StorageClass>(words[at + 2u]) == spv::StorageClassWorkgroup) {
				wg_ptrs[words[at + 1u]] = words[at + 3u];
			} else if (op == spv::OpVariable && count >= 4u &&
			           static_cast<spv::StorageClass>(words[at + 3u]) == spv::StorageClassWorkgroup) {
				if (const auto it = wg_ptrs.find(words[at + 1u]); it != wg_ptrs.end()) {
					wg_var_types.insert(it->second);
				}
			}
			at += count;
		}

		bool found_spec_workgroup_array = false;
		for (const auto arr_id: wg_var_types) {
			const auto arr_it = array_types.find(arr_id);
			if (arr_it == array_types.end()) continue;
			const uint32_t len_id = arr_it->second.second;
			const auto     sid_it = spec_ids.find(len_id);
			const auto     def_it = spec_defaults.find(len_id);
			if (sid_it != spec_ids.end() &&
			    sid_it->second == Recompiler::Spirv::MeshAllocationLengthSpecId &&
			    def_it != spec_defaults.end() &&
			    def_it->second == Recompiler::Spirv::MeshAllocationBaseDwords) {
				found_spec_workgroup_array = true;
				break;
			}
		}
		Require(found_spec_workgroup_array,
		        "mesh SPIR-V missing Workgroup array sized by OpSpecConstant SpecId 0 = 2");

		const auto ordinary = CompileMeshWithLds(LdsDwords, subgroup, false);
		for (size_t i = 5; i < ordinary.spirv.size(); i += ordinary.spirv[i] >> spv::WordCountShift) {
			Require((ordinary.spirv[i] & spv::OpCodeMask) != spv::OpSpecConstant,
			        "unaffected profile specialized the mesh allocation");
		}
		const uint32_t expected_wg_bytes =
		    (LdsDwords + Recompiler::Spirv::MeshAllocationBaseDwords) * 4u;
		const uint32_t got_wg_bytes = Recompiler::Spirv::MeshWorkgroupBytes(words);
		Require(got_wg_bytes == expected_wg_bytes,
		        "MeshWorkgroupBytes mismatch: got " + std::to_string(got_wg_bytes) + ", expected " +
		            std::to_string(expected_wg_bytes));
	}

	const uint32_t aliased_bytes = Recompiler::Spirv::MeshWorkgroupBytes(AliasedWorkgroupSpirv);
	Require(aliased_bytes == 64u,
	        "Aliased Workgroup blocks should take max (64), got " + std::to_string(aliased_bytes));

	constexpr size_t CuLds  = 65536u;
	constexpr size_t WgpLds = CuLds * 2u;
	constexpr size_t MaxPad = 32768u;
	const auto       free_bytes = [&](size_t alloc) -> std::pair<size_t, size_t> {
		const size_t cu_waves  = std::min(CuLds / alloc, size_t {4u});
		const size_t wgp_waves = std::min(WgpLds / alloc, size_t {8u});
		return {CuLds - cu_waves * alloc, WgpLds - wgp_waves * alloc};
	};

	Require(MeshLdsWgpPadDwords(15872u, CuLds, MaxPad) == 256u, "16 KB granule should pad by 1 KB");
	Require(MeshLdsWgpPadDwords(17920u, CuLds, MaxPad) == 256u, "18 KB granule should pad by 1 KB");
	Require(MeshLdsWgpPadDwords(20992u, CuLds, MaxPad) == 256u, "21 KB granule should pad by 1 KB");
	Require(MeshLdsWgpPadDwords(25088u, CuLds, MaxPad) == 256u, "25 KB granule should pad by 1 KB");
	Require(MeshLdsWgpPadDwords(31744u, CuLds, MaxPad) == 512u, "31 KB granule should pad by 2 KB");
	Require(MeshLdsWgpPadDwords(32768u, CuLds, MaxPad) == 256u, "32 KB granule should pad by 1 KB");
	Require(MeshLdsWgpPadDwords(41984u, CuLds, MaxPad) == 512u, "41 KB granule should pad by 2 KB");
	Require(MeshLdsWgpPadDwords(43008u, CuLds, MaxPad) == 256u, "42 KB granule should pad by 1 KB");
	Require(MeshLdsWgpPadDwords(18944u, CuLds, MaxPad) == 0u, "19 KB granule should not pad");
	Require(MeshLdsWgpPadDwords(24064u, CuLds, MaxPad) == 0u, "24 KB granule should not pad");
	Require(MeshLdsWgpPadDwords(0u, CuLds, MaxPad) == 0u, "zero LDS should not pad");
	Require(MeshLdsWgpPadDwords(17920u, 0u, MaxPad) == 0u, "zero CU LDS should not pad");
	Require(!MeshLdsWgpPadDwords(17920u, CuLds, 512u), "insufficient max_pad should not pad");
	Require(!MeshLdsWgpPadDwords(31744u, CuLds, 1024u), "31 KB with 1 KB cap should not pad");
	Require(!MeshLdsWgpPadDwords(63488u, CuLds, MaxPad), "62 KB cannot reach a safe granule");

	Require(MeshLdsWgpPadDwords(61u * 1024u, CuLds, MaxPad) == 0u, "61 KB measured safe at the CU limit");

	for (size_t kb = 1; kb <= 60; kb++) {
		const size_t   base = kb * 1024u;
		const auto result = MeshLdsWgpPadDwords(base, CuLds, MaxPad);
		Require(result.has_value(), "no usable allocation for bucket " + std::to_string(kb));
		const uint32_t pad = *result;
		const auto [cu_free, wgp_free] = free_bytes(base + static_cast<size_t>(pad) * 4u);
		Require(cu_free >= 4096u && wgp_free >= 8192u,
		        "bucket " + std::to_string(kb) + " KB after pad=" + std::to_string(pad) +
		            " left cu_free=" + std::to_string(cu_free) + " wgp_free=" + std::to_string(wgp_free));
		if (pad != 0u) {
			const size_t bytes = static_cast<size_t>(pad) * sizeof(uint32_t);
			Require(MeshLdsWgpPadDwords(base, CuLds, bytes) == pad &&
			            !MeshLdsWgpPadDwords(base, CuLds, bytes - 1u),
			        "incorrect padding at the shared-memory limit for bucket " + std::to_string(kb));
		}
	}
}

int DrawMeshLdsWgpPad() {
	const vk::detail::DynamicLoader loader;
	if (!loader.success()) return NoDevice("the Vulkan loader is missing");
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) return NoDevice("no Vulkan 1.3 instance");
	const auto instance = created.value;
	dispatch.init(instance);

	vk::PhysicalDevice                         physical;
	vk::PhysicalDeviceProperties               physical_props {};
	vk::PhysicalDeviceMeshShaderPropertiesEXT  selected_mesh_props {};
	uint32_t                                   family                 = 0;
	uint32_t                                   subgroup               = 0;
	bool                                       has_amd_shader_info    = false;
	bool                                       has_cache_control      = false;
	bool                                       skipped_llvmpipe       = false;
	bool                                       lacking_device_features = false;
	std::string                                llvmpipe_info;

	for (const auto candidate: Checked(instance.enumeratePhysicalDevices(), "device enumeration")) {
		vk::PhysicalDeviceDriverProperties driver_props;
		vk::PhysicalDeviceProperties2      base_props;
		base_props.pNext = &driver_props;
		candidate.getProperties2(&base_props);
		if (driver_props.driverID == vk::DriverId::eMesaLlvmpipe &&
		    base_props.properties.driverVersion <= VK_MAKE_API_VERSION(0, 26, 2, 4)) {
			skipped_llvmpipe = true;
			llvmpipe_info    = driver_props.driverInfo.data();
			continue;
		}
		const auto extensions = Checked(candidate.enumerateDeviceExtensionProperties(nullptr),
		                                "device extension enumeration");
		const auto has_ext    = [&](const char* ext) {
			return std::ranges::any_of(extensions, [ext](const auto& props) {
				return std::string_view(props.extensionName.data()) == ext;
			});
		};
		if (!has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME) ||
		    !has_ext(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME) ||
		    !has_ext(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME)) {
			lacking_device_features = true;
			continue;
		}
		vk::PhysicalDeviceMeshShaderFeaturesEXT       mesh_features;
		vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control;
		vk::PhysicalDeviceDepthClipEnableFeaturesEXT  depth_clip_enable;
		vk::PhysicalDeviceVulkan12Features            v12_features;
		vk::PhysicalDeviceVulkan13Features            v13_features;
		vk::PhysicalDeviceFeatures2                   features;
		v13_features.pNext       = &v12_features;
		v12_features.pNext       = &mesh_features;
		mesh_features.pNext      = &depth_clip_control;
		depth_clip_control.pNext = &depth_clip_enable;
		features.pNext           = &v13_features;
		candidate.getFeatures2(&features);

		const auto families = candidate.getQueueFamilyProperties();
		const auto graphics = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eGraphics);
		});
		if (!mesh_features.meshShader || !depth_clip_control.depthClipControl ||
		    !depth_clip_enable.depthClipEnable || !v12_features.bufferDeviceAddress ||
		    !v12_features.uniformBufferStandardLayout || !v13_features.dynamicRendering ||
		    !features.features.shaderInt64 || !features.features.depthClamp ||
		    !features.features.depthBounds || graphics == families.end()) {
			lacking_device_features = true;
			continue;
		}
		vk::PhysicalDeviceVulkan11Properties      props11;
		vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_props;
		vk::PhysicalDeviceProperties2             props;
		props11.pNext = &mesh_props;
		props.pNext   = &props11;
		candidate.getProperties2(&props);
		if (mesh_props.maxMeshWorkGroupInvocations < 64u ||
		    mesh_props.maxMeshWorkGroupSize[0] < 64u || mesh_props.maxMeshOutputVertices < 64u ||
		    mesh_props.maxMeshOutputPrimitives < 1u || mesh_props.maxMeshSharedMemorySize < 16384u ||
		    props11.subgroupSize == 0u) {
			lacking_device_features = true;
			continue;
		}
		const bool software = props.properties.deviceType == vk::PhysicalDeviceType::eCpu;
		if (physical && software) continue;
		physical            = candidate;
		physical_props      = props.properties;
		selected_mesh_props = mesh_props;
		family              = static_cast<uint32_t>(graphics - families.begin());
		subgroup            = props11.subgroupSize;
		has_amd_shader_info = KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS &&
		                      NeedsMeshLdsPadding(physical_props.vendorID, physical_props.deviceID, driver_props.driverID) &&
		                      has_ext(VK_AMD_SHADER_INFO_EXTENSION_NAME);
		has_cache_control   = v13_features.pipelineCreationCacheControl == VK_TRUE;
		if (!software) break;
	}
	if (!physical) {
		instance.destroy();
		if (skipped_llvmpipe) {
			std::printf("%s_draw: skipped, the device has no mesh shader compiler that handles "
			            "these programs (llvmpipe %s)\n",
			            Name, llvmpipe_info.c_str());
			return 77;
		}
		if (lacking_device_features) {
			std::printf("%s_draw: skipped, the device lacks VK_EXT_mesh_shader, depth clip "
			            "extensions, or depthBounds\n",
			            Name);
			return 77;
		}
		return NoDevice("no device with VK_EXT_mesh_shader, depth clip extensions, and dynamic rendering");
	}

	std::vector<const char*> enabled_extensions {VK_EXT_MESH_SHADER_EXTENSION_NAME,
	                                             VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	                                             VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME};
	if (has_amd_shader_info) {
		enabled_extensions.push_back(VK_AMD_SHADER_INFO_EXTENSION_NAME);
	}
	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;

	vk::PhysicalDeviceMeshShaderFeaturesEXT       mesh_features;
	mesh_features.meshShader = VK_TRUE;
	vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control;
	depth_clip_control.depthClipControl = VK_TRUE;
	vk::PhysicalDeviceDepthClipEnableFeaturesEXT  depth_clip_enable;
	depth_clip_enable.depthClipEnable = VK_TRUE;
	vk::PhysicalDeviceVulkan12Features v12_features;
	v12_features.bufferDeviceAddress         = VK_TRUE;
	v12_features.uniformBufferStandardLayout = VK_TRUE;
	vk::PhysicalDeviceVulkan13Features v13_features;
	v13_features.dynamicRendering             = VK_TRUE;
	v13_features.pipelineCreationCacheControl = has_cache_control ? VK_TRUE : VK_FALSE;
	v13_features.pNext                        = &v12_features;
	v12_features.pNext                        = &mesh_features;
	mesh_features.pNext                       = &depth_clip_control;
	depth_clip_control.pNext                  = &depth_clip_enable;
	vk::PhysicalDeviceFeatures2 features;
	features.features.shaderInt64 = VK_TRUE;
	features.features.depthClamp  = VK_TRUE;
	features.features.depthBounds = VK_TRUE;
	features.pNext                = &v13_features;

	vk::DeviceCreateInfo device_info;
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(enabled_extensions.size());
	device_info.ppEnabledExtensionNames = enabled_extensions.data();
	const auto device                   = Checked(physical.createDevice(device_info), "device creation");
	dispatch.init(device);

	vk::PhysicalDeviceDriverProperties driver;
	vk::PhysicalDeviceProperties2      properties;
	properties.pNext = &driver;
	physical.getProperties2(&properties);
	const auto gpu_name = std::string(properties.properties.deviceName.data()) + " (" +
	                      driver.driverName.data() + ", " + driver.driverInfo.data() + ")";
	const auto queue = device.getQueue(family, 0);

	GraphicContext graphics;
	graphics.instance                      = instance;
	graphics.physical_device               = physical;
	graphics.physical_device_properties    = physical_props;
	graphics.device                        = device;
	graphics.queue_family                  = family;
	graphics.queue                         = queue;
	graphics.mesh_shader_enabled           = true;
	graphics.mesh_shader_properties        = selected_mesh_props;
	graphics.pipeline_cache_lookup_enabled = has_cache_control;
	if (has_amd_shader_info) {
		graphics.fn_get_shader_info =
		    reinterpret_cast<PFN_vkGetShaderInfoAMD>(device.getProcAddr("vkGetShaderInfoAMD"));
		Require(graphics.fn_get_shader_info != nullptr, "AMD shader statistics entry point missing");
	}

	constexpr uint32_t Side   = 16u;
	constexpr uint32_t Pixels = Side * Side;
	constexpr uint32_t Bytes  = Pixels * 4u;
	constexpr auto     Format = vk::Format::eR8G8B8A8Unorm;

	ShaderModule pixel_mod {};
	pixel_mod.info.stage       = ShaderType::Pixel;
	pixel_mod.info.shader_hash = 0x706978656c000001ull;
	{
		vk::ShaderModuleCreateInfo info;
		info.codeSize    = InterpPixelSpirv.size() * sizeof(uint32_t);
		info.pCode       = InterpPixelSpirv.data();
		pixel_mod.module = Checked(device.createShaderModule(info), "pixel shader module");
	}

	GraphicsPipelineKey key {};
	key.rendering.color_count                  = 1u;
	key.rendering.color_formats[0]             = Format;
	key.rendering.depth_format                 = vk::Format::eUndefined;
	key.rendering.stencil_format               = vk::Format::eUndefined;
	key.static_params.samples                  = 1u;
	key.static_params.polygon_mode             = vk::PolygonMode::eFill;
	key.static_params.depth_clip_enable        = true;
	key.static_params.color_mask[0]            = 0x0fu;
	key.static_params.color_srcblend[0]        = static_cast<uint32_t>(Prospero::BlendFactor::kOne);
	key.static_params.color_destblend[0]       = static_cast<uint32_t>(Prospero::BlendFactor::kZero);
	key.static_params.color_comb_fcn[0]        = static_cast<uint32_t>(Prospero::BlendOp::kAdd);

	const auto make_mesh_module = [&](uint32_t lds_dwords) -> ShaderModule {
		auto         compiled = CompileMeshWithLds(lds_dwords, subgroup, graphics.fn_get_shader_info != nullptr);
		ShaderModule mod {};
		mod.info.stage        = ShaderType::Mesh;
		mod.info.shader_hash  = 0x6d6573686c647300ull ^ lds_dwords;
		mod.mesh_shared_bytes = Recompiler::Spirv::MeshWorkgroupBytes(compiled.spirv);
		vk::ShaderModuleCreateInfo info;
		info.codeSize = compiled.spirv.size() * sizeof(uint32_t);
		info.pCode    = compiled.spirv.data();
		mod.module    = Checked(device.createShaderModule(info), "mesh shader module");
		return mod;
	};

	const auto probe_raw_lds = [&](const ShaderModule& mesh_mod, vk::PipelineCache cache) {
		const std::array<const ShaderModule*, 4> stages {&mesh_mod, nullptr, nullptr, &pixel_mod};
		PipelineCache::Pipeline layout {};
		layout.native_bindings = true;
		CreateGraphicsPipelineLayout(graphics, layout, key, stages);
		const auto query = std::exchange(graphics.fn_get_shader_info, nullptr);
		vk::Pipeline pipe;
		Checked(CreateGraphicsPipelineWhole(graphics, layout, key, stages, cache, {}, pipe), "probe pipeline");
		graphics.fn_get_shader_info = query;
		VkShaderStatisticsInfoAMD stats {};
		size_t size = sizeof(stats);
		Require(query(device, pipe, VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_INFO_TYPE_STATISTICS_AMD,
		              &size, &stats) == VK_SUCCESS && size == sizeof(stats), "mesh statistics unavailable");
		const auto pad = QueryMeshLdsWgpPadDwords(graphics, pipe, mesh_mod);
		device.destroyPipeline(pipe);
		device.destroyPipelineLayout(layout.pipeline_layout);
		device.destroyDescriptorSetLayout(layout.descriptor_set_layout);
		return std::pair {pad.value_or(0), stats.resourceUsage.ldsUsageSizeInBytes};
	};

	ShaderModule mesh_mod {};
	uint32_t     probe_pad = 0;
	size_t       probe_lds = 0;
	if (graphics.fn_get_shader_info) {
		for (const uint32_t candidate_dwords: {3200u, 3328u, 3456u, 3584u, 3712u, 3840u, 3968u, 4096u}) {
			auto candidate = make_mesh_module(candidate_dwords);
			const auto [pad, lds] = probe_raw_lds(candidate, nullptr);
			if (pad != 0u) {
				mesh_mod  = std::move(candidate);
				probe_pad = pad;
				probe_lds = lds;
				break;
			}
			device.destroyShaderModule(candidate.module);
		}
		Require(mesh_mod.module != nullptr && probe_pad != 0u,
		        "could not synthesize an unsafe WGP LDS bucket on AMD device");
	} else {
		mesh_mod = make_mesh_module(3200u);
	}

	const std::array<const ShaderModule*, 4> stages {&mesh_mod, nullptr, nullptr, &pixel_mod};
	PipelineCache::Pipeline                  pipeline {};
	pipeline.native_bindings = true;
	CreatePipelineInternal(graphics, pipeline, key, stages, nullptr);
	Require(pipeline.pipeline != nullptr, "CreatePipelineInternal returned null pipeline");

	size_t padded_lds = 0;
	if (graphics.fn_get_shader_info) {
		const auto remaining = QueryMeshLdsWgpPadDwords(graphics, pipeline.pipeline, mesh_mod);
		VkShaderStatisticsInfoAMD stats {};
		size_t                    stats_size = sizeof(stats);
		Require(graphics.fn_get_shader_info(device, pipeline.pipeline, VK_SHADER_STAGE_MESH_BIT_EXT,
		                                    VK_SHADER_INFO_TYPE_STATISTICS_AMD, &stats_size,
		                                    &stats) == VK_SUCCESS,
		        "vkGetShaderInfoAMD failed on built pipeline");
		padded_lds = stats.resourceUsage.ldsUsageSizeInBytes;
		Require(remaining == 0u,
		        "built mesh pipeline still in unsafe WGP LDS bucket: lds=" +
		            std::to_string(padded_lds) + " remaining_pad=" + std::to_string(remaining.value_or(UINT32_MAX)));
		Require(padded_lds >= probe_lds + static_cast<size_t>(probe_pad) * 4u,
		        "built mesh pipeline LDS did not grow as expected: probe=" +
		            std::to_string(probe_lds) + " padded=" + std::to_string(padded_lds));
	}

#if defined(ANYPS5_KYTY)
	if (graphics.fn_get_shader_info && has_cache_control) {
		vk::PipelineCacheCreateInfo cache_info {};
		const auto vk_cache = Checked(device.createPipelineCache(cache_info), "pipeline cache");
		(void)probe_raw_lds(mesh_mod, vk_cache);
		const auto check_lookup = [&] {
			vk::Pipeline cached = nullptr;
			const auto result = CreateGraphicsPipelineWhole(graphics, pipeline, key, stages, vk_cache,
			                       vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired, cached);
			if (result == vk::Result::ePipelineCompileRequired) {
				Require(!cached, "cache miss returned a pipeline");
			} else {
				Require(result == vk::Result::eSuccess && cached &&
				            QueryMeshLdsWgpPadDwords(graphics, cached, mesh_mod) == 0u,
				        "cache hit did not return a safe mesh pipeline");
				device.destroyPipeline(cached);
			}
		};
		check_lookup();

		vk::Pipeline built_pipe = nullptr;
		const auto   built_res =
		    CreateGraphicsPipelineWhole(graphics, pipeline, key, stages, vk_cache, {}, built_pipe);
		Require(built_res == vk::Result::eSuccess && built_pipe,
		        "CreateGraphicsPipelineWhole failed to build padded pipeline into cache");
		Require(QueryMeshLdsWgpPadDwords(graphics, built_pipe, mesh_mod) == 0u,
		        "cached padded pipeline still unsafe");
		device.destroyPipeline(built_pipe);

		check_lookup();
		device.destroyPipelineCache(vk_cache);
	}
#endif

	const auto memory_types = physical.getMemoryProperties();
	const auto allocate     = [&](const vk::MemoryRequirements& needs, vk::MemoryPropertyFlags flags) {
		for (uint32_t type = 0; type < memory_types.memoryTypeCount; type++) {
			if (((needs.memoryTypeBits >> type) & 1u) == 0u ||
			    (memory_types.memoryTypes[type].propertyFlags & flags) != flags) {
				continue;
			}
			vk::MemoryAllocateInfo info;
			info.allocationSize  = needs.size;
			info.memoryTypeIndex = type;
			return Checked(device.allocateMemory(info), "memory allocation");
		}
		Require(false, "no suitable memory type");
		return vk::DeviceMemory {};
	};

	vk::ImageCreateInfo image_info;
	image_info.imageType   = vk::ImageType::e2D;
	image_info.format      = Format;
	image_info.extent      = vk::Extent3D {Side, Side, 1u};
	image_info.mipLevels   = 1u;
	image_info.arrayLayers = 1u;
	image_info.samples     = vk::SampleCountFlagBits::e1;
	image_info.tiling      = vk::ImageTiling::eOptimal;
	image_info.usage       = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
	const auto image        = Checked(device.createImage(image_info), "image creation");
	const auto image_memory = allocate(device.getImageMemoryRequirements(image),
	                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
	Checked(device.bindImageMemory(image, image_memory, 0), "image bind");

	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u};
	vk::ImageViewCreateInfo         view_info;
	view_info.image            = image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = Format;
	view_info.subresourceRange = whole;
	const auto view            = Checked(device.createImageView(view_info), "image view creation");

	vk::BufferCreateInfo buffer_info;
	buffer_info.size  = Bytes;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	const auto buffer = Checked(device.createBuffer(buffer_info), "readback buffer creation");
	const auto buffer_memory =
	    allocate(device.getBufferMemoryRequirements(buffer),
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	Checked(device.bindBufferMemory(buffer, buffer_memory, 0), "buffer bind");

	vk::CommandPoolCreateInfo pool_info;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = family;
	const auto pool            = Checked(device.createCommandPool(pool_info), "command pool");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = pool;
	command_info.commandBufferCount = 1;
	const auto command =
	    Checked(device.allocateCommandBuffers(command_info), "command buffer").front();
	const auto fence = Checked(device.createFence(vk::FenceCreateInfo {}), "fence creation");

	const auto image_barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags source,
	                               vk::AccessFlags target, vk::PipelineStageFlags before,
	                               vk::PipelineStageFlags after) {
		vk::ImageMemoryBarrier barrier;
		barrier.srcAccessMask       = source;
		barrier.dstAccessMask       = target;
		barrier.oldLayout           = from;
		barrier.newLayout           = to;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image;
		barrier.subresourceRange    = whole;
		command.pipelineBarrier(before, after, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	};

	Checked(command.begin(vk::CommandBufferBeginInfo {}), "command begin");
	image_barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
	              vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
	              vk::PipelineStageFlagBits::eColorAttachmentOutput);

	vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(Side), static_cast<float>(Side), 0.0f, 1.0f};
	vk::Rect2D   scissor {{0, 0}, {Side, Side}};
	vk::RenderingAttachmentInfo target;
	target.imageView   = view;
	target.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	target.loadOp      = vk::AttachmentLoadOp::eClear;
	target.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo pass;
	pass.renderArea           = scissor;
	pass.layerCount           = 1u;
	pass.colorAttachmentCount = 1u;
	pass.pColorAttachments    = &target;

	command.beginRendering(&pass);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	command.setViewportWithCount(1u, &viewport);
	command.setScissorWithCount(1u, &scissor);
	command.setLineWidth(1.0f);
	command.setCullMode(vk::CullModeFlagBits::eNone);
	command.setFrontFace(vk::FrontFace::eCounterClockwise);
	command.setDepthTestEnable(VK_FALSE);
	command.setDepthWriteEnable(VK_FALSE);
	command.setDepthCompareOp(vk::CompareOp::eAlways);
	command.setDepthBiasEnable(VK_FALSE);
	command.setDepthBias(0.0f, 0.0f, 0.0f);
	command.setStencilTestEnable(VK_FALSE);
	command.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, vk::StencilOp::eKeep,
	                     vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
	command.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xffu);
	command.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0u);
	command.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xffu);
	const float blend_constants[4] {0.0f, 0.0f, 0.0f, 0.0f};
	command.setBlendConstants(blend_constants);
#if !defined(__APPLE__)
	command.setDepthBoundsTestEnable(VK_FALSE);
	command.setDepthBounds(0.0f, 1.0f);
#endif
	std::array<uint32_t, Recompiler::IR::NativePushConstantSize / 4u> push {};
	push[0] = 3u;
	command.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
	                      Recompiler::IR::NativePushConstantSize, push.data());
	command.drawMeshTasksEXT(1u, 1u, 1u);
	command.endRendering();

	image_barrier(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
	              vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
	              vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer);
	vk::BufferImageCopy copy;
	copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0u, 0u, 1u};
	copy.imageExtent      = vk::Extent3D {Side, Side, 1u};
	command.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &copy);

	vk::BufferMemoryBarrier host_barrier;
	host_barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	host_barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.buffer              = buffer;
	host_barrier.offset              = 0;
	host_barrier.size                = Bytes;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &host_barrier, 0, nullptr);
	Checked(command.end(), "command end");

	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Checked(queue.submit(1, &submit, fence), "queue submit");
	Checked(device.waitForFences(1, &fence, VK_TRUE, 5000000000ull), "waitForFences");

	const auto* bytes =
	    static_cast<const uint8_t*>(Checked(device.mapMemory(buffer_memory, 0, Bytes), "readback map"));
	uint32_t matched = 0;
	for (uint32_t i = 0; i < Pixels; i++) {
		if (bytes[i * 4u + 0u] == 255u && bytes[i * 4u + 1u] == 0u &&
		    bytes[i * 4u + 2u] == 255u && bytes[i * 4u + 3u] == 255u) {
			matched++;
		}
	}
	device.unmapMemory(buffer_memory);
	Require(matched == Pixels,
	        "expected " + std::to_string(Pixels) + " magenta pixels, got " + std::to_string(matched));

	device.destroyFence(fence);
	device.destroyCommandPool(pool);
	device.destroyBuffer(buffer);
	device.freeMemory(buffer_memory);
	device.destroyImageView(view);
	device.destroyImage(image);
	device.freeMemory(image_memory);
	device.destroyPipeline(pipeline.pipeline);
	device.destroyPipelineLayout(pipeline.pipeline_layout);
	device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
	device.destroyShaderModule(pixel_mod.module);
	device.destroyShaderModule(mesh_mod.module);
	device.destroy();
	instance.destroy();

	std::printf("%s_draw: ok, subgroup=%u amd_shader_info=%s lds=%zu->%zu pad=%u on %s\n", Name,
	            subgroup, graphics.fn_get_shader_info ? "on" : "off", probe_lds, padded_lds,
	            probe_pad, gpu_name.c_str());
	return 0;
}

}

extern "C" int RunKytyMeshLdsWgpPadTest(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "draw") == 0) return DrawMeshLdsWgpPad();
	VerifyCodegenAndSizing();
	std::printf("%s: ok\n", Name);
	return 0;
}
