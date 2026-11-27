set -euo pipefail

if [[ $# != 5 ]]; then
    echo "usage: bash build.sh source-tree configured-build precache-snapshot headless-driver-source output-directory" >&2
    exit 2
fi
source_tree=$(realpath "$1")
configured=$(realpath "$2")
snapshot=$(realpath "$3")
driver=$(realpath "$4")
mkdir -p "$5"
output=$(realpath "$5")
probe=$(realpath "$(dirname "$0")/Replay.cpp")
cd "$output"

python3 - "$driver" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
marker = "struct PipelineCacheTestAccess {"
if source.count(marker) != 1:
    raise SystemExit("unsupported headless driver adapter")
host = source.split(marker)[0]
for variable, value in (("g_title", "offline"), ("g_version", "1")):
    declaration = f"std::string {variable};"
    if host.count(declaration) != 1:
        raise SystemExit(f"missing headless driver declaration for {variable}")
    host = host.replace(declaration, f'std::string {variable} = "{value}";')
Path("Host.cpp").write_text(host + "}\n")
PY

includes=(-I"$source_tree/core/libs" -I"$source_tree/core/kyty"
    -I"$source_tree/core/libs/prx/libSceAgcDriver/KytyShim/include" -I"$snapshot"
    -I"$configured/core/kyty/generated" -I"$configured/core/libs/prx/libSceAgcDriver/generated")
for path in zstd/lib SPIRV-Tools/include Vulkan-Headers/include VulkanMemoryAllocator/include SPIRV-Headers/include fmt/include xxHash xbyak magic_enum/include/magic_enum tracy/public; do
    includes+=(-isystem "$source_tree/3rdparty/$path")
done
flags=(-O2 -DNDEBUG -std=c++20 -fPIE -DANYPS5_KYTY=1 -DAPS5_ENABLE_TIMING_LOG=0
    -DTRACY_DELAYED_INIT -DTRACY_ENABLE -DTRACY_MANUAL_LIFETIME -DTRACY_NO_CRASH_HANDLER
    -DTRACY_ON_DEMAND -DXBYAK_NO_EXCEPTION)
for source in Host.cpp "$probe"; do
    "${CXX:-g++-14}" "${flags[@]}" "${includes[@]}" -c "$source" -o "$(basename "$source" .cpp).o"
done
cp "$snapshot/libc.prx" .
runtime=(libc.prx)
if [[ -f "$snapshot/libkernel.prx" ]]; then
    cp "$snapshot/libkernel.prx" .
    runtime+=(libkernel.prx)
fi
"${CXX:-g++-14}" -Wl,--gc-sections -Wl,--no-as-needed -fPIE -pie Host.o Replay.o \
    "$snapshot/obj/Config.o" "$snapshot/obj/NativeRt.o" "$snapshot/obj/Log.o" \
    '-Wl,-rpath,$ORIGIN' "${runtime[@]}" "$snapshot/libkyty_gpu.a" -ldl -lpthread -o graphics-replay
