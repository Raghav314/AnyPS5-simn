# kyty

## add Functional

- [Scalar buffer load offsets](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterMemory.cpp) have not been measured on PS5 hardware when the low bits of `SOFFSET` and the instruction immediate would carry into the next dword. The implementation masks each operand before addition, matching the [Mesa ACO SMEM rule](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/fff0e3f85c4fbf87bc7cfe5be2544e831c584ca1/src/amd/compiler/aco_optimizer.cpp#1905) and the RDNA2 measurements in [issue #216](https://github.com/simnJS/AnyPS5/issues/216). The RDNA2 ISA's buffered-address pseudocode suggests a final mask instead. Synthetic replay covers carry, no-carry and a crossed bound. Closes when a PS5 hardware oracle measures the cases, including a multi-dword load at the bound.
