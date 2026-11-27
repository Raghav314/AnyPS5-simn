# kyty

## add Functional

- [Indexed buffer reads](../../../core/kyty/graphics/host_gpu/renderer/renderContext.cpp) can still return zero on a cold device-address page when the selector data was last written by the GPU, a child page aliases a GPU-written image, a completion label is pending, the shader derives an address through an unrecognized expression, or the required reads exceed the renderer's probe or page budget. Closes when first reads from these paths see the current buffer contents without an earlier binding on the same page.
