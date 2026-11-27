# kyty

## add Functional

- [KYTY_RT_STATS](../../../core/kyty/graphics/host_gpu/renderer/pipeline/pipelineShaderModule.cpp) instruments compute shaders and selected pixel shaders during native ray tracing. Its GPU measurements use different modules than normal runs and cannot be compared directly with them. Closes when the statistics can be collected without changing compiled shader modules.
