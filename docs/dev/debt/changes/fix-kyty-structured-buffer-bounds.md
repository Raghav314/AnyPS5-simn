# kyty

## add Functional

- [Direct buffer access](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterMemory.cpp) bounds mode 0 structured accesses within a record and mode 3 accesses by complete payload elements. The Vulkan descriptor range can still clip mode 0 accesses shifted by `soffset`, and modes 1 and 2 can require addresses beyond that range. The RDNA2 ISA does not specify carry handling when the lane ID is added to the index or the instruction immediate is added to the VGPR offset; mode 0 currently drops an access on carry pending native hardware measurement.
