# kyty

## add Functional

- [Direct buffer access](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterMemory.cpp) bounds mode 0 structured accesses within a record and mode 3 accesses by complete payload elements. The Vulkan descriptor range can still clip mode 0 accesses shifted by `soffset`, and modes 1 and 2 can require addresses beyond that range.
