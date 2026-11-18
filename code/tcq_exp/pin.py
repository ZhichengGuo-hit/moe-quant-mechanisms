"""CPU 亲和性绑定 (Windows 双路 Xeon, 2 NUMA node)。

所有计时/生成实验必须先绑定, 否则同形状跨 run 波动 20-50%。

用法:
    python pin.py topology                     # 打印 NUMA 节点与逻辑 CPU 分布
    python pin.py run --numa 0 -- <cmd...>     # 绑定 node0 全部逻辑 CPU 后启动子进程
    python pin.py run --cpus 0-7 -- <cmd...>   # 绑定 8 个逻辑 CPU
    python pin.py bench                        # 绑定方案对照 (8/16/跨 node)

作为库:
    from pin import pin_self, numa_cpus
    pin_self(numa=0)        # 当前进程绑定 node0
"""
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def numa_masks():
    """返回 {node: 逻辑 CPU 列表}; 用 GetNumaNodeProcessorMask (<=64 逻辑 CPU)。"""
    highest = wt.ULONG(0)
    if not kernel32.GetNumaHighestNodeNumber(ctypes.byref(highest)):
        raise OSError(ctypes.get_last_error(), "GetNumaHighestNodeNumber")
    out = {}
    for node in range(highest.value + 1):
        mask = ctypes.c_ulonglong(0)
        ok = kernel32.GetNumaNodeProcessorMask(node, ctypes.byref(mask))
        if not ok:
            continue
        bits = [i for i in range(64) if mask.value >> i & 1]
        if bits:
            out[node] = bits
    return out


def numa_cpus(numa=0):
    m = numa_masks()
    if numa not in m:
        raise SystemExit(f"NUMA node {numa} 不存在, 可用: {sorted(m)}")
    return m[numa]


def parse_cpus(spec):
    """"0-7,16" -> [0..7, 16]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def mask_of(cpus):
    m = 0
    for c in cpus:
        m |= 1 << c
    return m


def set_affinity(cpus):
    m = mask_of(cpus)
    try:
        import psutil
        psutil.Process().cpu_affinity(cpus)
        return m
    except Exception:
        pass
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wt.BOOL
    h = kernel32.GetCurrentProcess()
    if not kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(m)):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    return m


def get_affinity():
    import psutil
    return psutil.Process().cpu_affinity()


def pin_self(numa=None, cpus=None):
    """在当前进程上绑定; 之后创建的子进程自动继承该亲和性。"""
    if cpus is None:
        cpus = numa_cpus(0 if numa is None else numa)
    elif isinstance(cpus, str):
        cpus = parse_cpus(cpus)
    set_affinity(cpus)
    return cpus


def topology():
    total = os.cpu_count()
    m = numa_masks()
    print(f"逻辑 CPU 总数: {total}")
    for node, cpus in sorted(m.items()):
        print(f"NUMA node {node}: {len(cpus)} 逻辑 CPU -> {cpus}")
    return m


def run_child(argv, cpus):
    pin_self(cpus=cpus)
    print(f"[pin] 父进程绑定 {len(cpus)} CPU: {cpus}", file=sys.stderr, flush=True)
    rc = subprocess.call(argv)
    return rc


def _bench():
    """对照 8 逻辑(单 node 低半) / 16 逻辑(单 node) / 跨 node 各 8, 各 3 次空转。"""
    import time

    def spin(n):
        x = 0
        x = 0
        for _ in range(n):
            x += 1
        return x

    n = 30_000_000
    total = os.cpu_count()
    m = numa_masks()
    cpus_a = m[min(m)]
    plans = {
        "node0_all": cpus_a,
        "node0_half": cpus_a[: len(cpus_a) // 2],
        "node1_all": m[max(m)],
        "cross_8_8": cpus_a[: len(cpus_a) // 2] + m[max(m)][: len(m[max(m)]) // 2],
    }
    import time
    for name, cpus in plans.items():
        set_affinity(cpus)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            spin(n)
            ts.append(time.perf_counter() - t0)
        print(f"{name:12s} n={len(cpus):2d} best={min(ts)*1000:7.1f}ms "
              f"median={sorted(ts)[1]*1000:7.1f}ms var={(max(ts)-min(ts))/min(ts)*100:5.1f}%")
    set_affinity(list(range(total)))


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "topology":
        topology()
        return 0
    if cmd == "bench":
        _bench()
        return 0
    if cmd == "run":
        rest = argv[1:]
        cpus, numa = None, None
        i = 0
        while i < len(rest):
            if rest[i] == "--cpus":
                cpus = parse_cpus(rest[i + 1]); i += 2
            elif rest[i] == "--numa":
                numa = int(rest[i + 1]); i += 2
            elif rest[i] == "--":
                i += 1; break
            else:
                break
        return run_child(rest[i:], cpus if cpus is not None else numa_cpus(numa or 0))
    print(f"未知命令 {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
