#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr std::array<uint32_t, 7> FormattedWords {
    0x34020082u, 0xe0001000u, 0x80000201u, 0xbf8c0000u, 0xe0701000u, 0x80010201u, 0xbf810000u,
};

constexpr std::array<uint32_t, 2> QueueDependentWords {
    0xb9031818u, 0xbf810000u,
};

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what);
	++g_failures;
}

ShaderComputeInputInfo BaseComputeInput() {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 32;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	info.workgroup_register = 8;
	info.group_id[0]        = true;
	return info;
}

void CheckFormattedBufferAndAsyncGating() {
	const auto                    base_input = BaseComputeInput();
	const std::vector<uint32_t>     user_data(base_input.workgroup_register, 0u);
	const std::vector<uint32_t>     code(FormattedWords.begin(), FormattedWords.end());
	auto                            input = base_input;
	Recompiler::CompileOptions      options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = input.wave_size;
	options.shader_hash        = 0x1111u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &input;

	const auto source          = Recompiler::PrepareShaderSource(code, options);
	const bool queue_dependent = Recompiler::Decoder::ComputeShaderDependsOnQueue(code);
	Check(!queue_dependent, "formatted buffer shader is not queue-dependent");
	auto       translated = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan       = IR::ExtractResourcePlan(translated.program);

	const auto fallback = SynthesizeComputeVariants(
	    base_input, queue_dependent, translated.program, plan, {});
	Check(!fallback.empty(), "fallback generates formatted buffer variants");
	Check(std::ranges::none_of(fallback, [](const ComputeRecipeVariant& v) { return v.input.async_compute; }),
	      "non-queue-dependent shader omits dead async_compute=true variants");

	const auto has_format = [&](Prospero::BufferFormat format) {
		return std::ranges::any_of(fallback, [&](const ComputeRecipeVariant& v) {
			return v.specialization.buffers.size() == 2u &&
			       v.specialization.buffers[0].descriptor_format == format;
		});
	};
	Check(has_format(Prospero::BufferFormat::k32UInt), "fallback covers k32UInt formatted buffers");
	Check(has_format(Prospero::BufferFormat::k32Float), "fallback retains k32Float formatted buffers");

	for (size_t i = 0; i < fallback.size(); ++i) {
		auto v_input               = fallback[i].input;
		options.input_info.compute = &v_input;
		auto       retranslated    = Recompiler::TranslateProgram(source.decoded, options);
		const auto compiled        = Recompiler::CompileProgram(std::move(retranslated), options, fallback[i].specialization);
		Check(!compiled.spirv.empty() && compiled.spirv[0] == 0x07230203u, "fallback variant compiles to SPIR-V");
	}
}

void CheckDonorSignaturesAndRejections() {
	const auto                    base_input = BaseComputeInput();
	const std::vector<uint32_t>     user_data(base_input.workgroup_register, 0u);
	const std::vector<uint32_t>     code(FormattedWords.begin(), FormattedWords.end());
	auto                            input = base_input;
	Recompiler::CompileOptions      options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = input.wave_size;
	options.shader_hash        = 0x2222u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &input;

	const auto source          = Recompiler::PrepareShaderSource(code, options);
	const bool queue_dependent = Recompiler::Decoder::ComputeShaderDependsOnQueue(code);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan            = IR::ExtractResourcePlan(translated.program);

	IR::ResourceSpecialization compatible_shape;
	compatible_shape.buffers.resize(2u);
	compatible_shape.buffers[0].descriptor_format = Prospero::BufferFormat::k32UInt;
	compatible_shape.buffers[0].packed_stride     = 4u;
	compatible_shape.buffers[1].descriptor_format = Prospero::BufferFormat::kInvalid;

	auto donor_input                       = base_input;
	donor_input.dispatch_thread_dimensions = true;
	donor_input.dispatch_indirect_threads  = true;
	donor_input.async_compute              = true;

	ManifestModule compatible_donor {
	    .stage           = ShaderType::Compute,
	    .code_hash       = 0x9999u,
	    .user_data_count = 8u,
	    .input           = donor_input,
	    .specialization  = compatible_shape,
	};

	ManifestModule wrong_stage_donor {
	    .stage           = ShaderType::Pixel,
	    .code_hash       = 0x8888u,
	    .user_data_count = 8u,
	    .input           = ShaderPixelInputInfo {},
	    .specialization  = compatible_shape,
	};

	auto wrong_count_shape = compatible_shape;
	wrong_count_shape.buffers.pop_back();
	ManifestModule wrong_count_donor {
	    .stage           = ShaderType::Compute,
	    .code_hash       = 0x7777u,
	    .user_data_count = 8u,
	    .input           = base_input,
	    .specialization  = wrong_count_shape,
	};

	auto wrong_indirect_shape = compatible_shape;
	wrong_indirect_shape.buffers[0].indirect_root = 0u;
	ManifestModule wrong_indirect_donor {
	    .stage           = ShaderType::Compute,
	    .code_hash       = 0x6666u,
	    .user_data_count = 8u,
	    .input           = base_input,
	    .specialization  = wrong_indirect_shape,
	};

	const std::array<ManifestModule, 4> donors {
	    wrong_stage_donor, wrong_count_donor, wrong_indirect_donor, compatible_donor,
	};
	const auto synthesized = SynthesizeComputeVariants(
	    base_input, queue_dependent, translated.program, plan, donors);
	Check(synthesized.size() == 1u, "accepts only the compatible buffer-only donor and rejects incompatible ones");
	if (synthesized.size() == 1u) {
		const auto& v = synthesized[0];
		Check(v.input.dispatch_thread_dimensions && v.input.dispatch_indirect_threads,
		      "preserves dispatch_thread_dimensions and dispatch_indirect_threads from donor");
		Check(!v.input.async_compute, "gates donor async_compute on compute_queue_dependent");
		Check(v.specialization.buffers[0].descriptor_format == Prospero::BufferFormat::k32UInt,
		      "preserves donor formatted buffer format");

		std::vector<uint32_t> expected_key;
		auto                  witness_input = base_input;
		witness_input.dispatch_thread_dimensions = true;
		witness_input.dispatch_indirect_threads  = true;
		BuildStageStaticKey(witness_input, expected_key);

		std::vector<uint32_t> synthesized_key;
		BuildStageStaticKey(v.input, synthesized_key);
		Check(synthesized_key == expected_key, "synthesized static key matches runtime witness key");

		PipelineManifest manifest;
		ManifestModule   entry {};
		entry.stage           = ShaderType::Compute;
		entry.code_hash       = 0x2222u;
		entry.user_data_count = 8u;
		entry.input           = v.input;
		entry.specialization  = v.specialization;
		manifest.modules.push_back(std::move(entry));
		const auto       bytes = SaveManifest(manifest);
		PipelineManifest loaded;
		Check(LoadManifest(bytes, loaded) && loaded.modules.size() == 1u, "manifest round-trip succeeds");
		if (loaded.modules.size() == 1u) {
			const auto* loaded_input = std::get_if<ShaderComputeInputInfo>(&loaded.modules[0].input);
			Check(loaded_input != nullptr && loaded_input->dispatch_thread_dimensions &&
			          loaded_input->dispatch_indirect_threads,
			      "manifest round-trip preserves dispatch_thread_dimensions and dispatch_indirect_threads");
		}
	}
}

void CheckQueueDependentVariants() {
	const auto                    base_input = BaseComputeInput();
	const std::vector<uint32_t>     user_data(base_input.workgroup_register, 0u);
	const std::vector<uint32_t>     code(QueueDependentWords.begin(), QueueDependentWords.end());
	auto                            input = base_input;
	Recompiler::CompileOptions      options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = input.wave_size;
	options.shader_hash        = 0x3333u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &input;

	const auto source          = Recompiler::PrepareShaderSource(code, options);
	const bool queue_dependent = Recompiler::Decoder::ComputeShaderDependsOnQueue(code);
	Check(queue_dependent, "queue register shader is queue-dependent");
	auto       translated = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan       = IR::ExtractResourcePlan(translated.program);
	const auto variants   = SynthesizeComputeVariants(
	    base_input, queue_dependent, translated.program, plan, {});
	const bool has_sync = std::ranges::any_of(variants, [](const ComputeRecipeVariant& v) { return !v.input.async_compute; });
	const bool has_async = std::ranges::any_of(variants, [](const ComputeRecipeVariant& v) { return v.input.async_compute; });
	Check(variants.size() == 2u && has_sync && has_async,
	      "queue-dependent shader emits both sync and async compute variants");
}

}

int main() {
	CheckFormattedBufferAndAsyncGating();
	CheckDonorSignaturesAndRejections();
	CheckQueueDependentVariants();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	return 0;
}
