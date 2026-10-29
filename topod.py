#!/usr/bin/env python3
"""Live data-flow topology of this machine.

Sources:
  eBPF (bpftrace, topo.bt)  block I/O per device, NIC bytes per interface, per-process attribution
  AMD data fabric PMU       DRAM channel traffic (perf_event_open, amd_df)
  NVML                      dGPU PCIe rx/tx + VRAM, only polled while the GPU is already awake
  sysfs / procfs            memory, zram, amdgpu, link speeds

Run as root:  sudo python3 topod.py   then open http://127.0.0.1:8787

Structure: one thread per data source, each writing its own key in the shared
`state` dict. A publisher thread wakes once a second and every connected
browser gets a JSON snapshot of `state` over Server-Sent Events. The sources
run at their own pace (eBPF 1 s, context 2 s); the page just shows the latest
value of each.
"""
import ctypes, json, os, re, struct, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get('PORT', 8787))
# Block device major numbers (see /proc/devices): 259 = blkext, used by NVMe; 253 = zram.
NVME_MAJOR, ZRAM_MAJOR = 259, 253
# PCI addresses from `lspci`; these are specific to this laptop.
DGPU_PCI = '0000:01:00.0'
NVME_PCI = '0000:04:00.0'

# Shared snapshot of everything we know, plus a condition variable that both
# guards it and lets the SSE handlers sleep until the next publish.
state = {}
cond = threading.Condition()


def read(path, default=None):
    """Read a small sysfs/procfs file, or return default if it's missing or unreadable."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


# ---------------------------------------------------------------- eBPF
#
# bpftrace does the kernel-side work (see topo.bt); we just run it as a child
# process and parse its JSON output. -B line makes it flush after every line
# even though stdout is a pipe, -q drops the "Attaching N probes" banner.

def ebpf_loop():
    proc = subprocess.Popen(['bpftrace', '-q', '-B', 'line', '-f', 'json', os.path.join(HERE, 'topo.bt')],
                            stdout=subprocess.PIPE, text=True)
    # One batch = the map lines between two "tick" markers. Collect them, then
    # turn byte counts into bytes/s using the actually-elapsed time, which is
    # never exactly 1 s.
    maps, last = {}, time.monotonic()
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get('type') == 'map':
            maps.update(msg['data'])         # e.g. {"@blk_wr": {"259": 741376}}
        elif msg.get('type') == 'printf' and msg.get('data') == 'tick':
            now = time.monotonic()
            dt, last = now - last, now
            rate = lambda name: {k: v / dt for k, v in maps.get(name, {}).items()}
            rd, wr = rate('@blk_rd'), rate('@blk_wr')
            # JSON object keys are strings, so the integer map keys arrive as "259".
            # Direction is from DRAM's point of view: a disk read brings data
            # "in" to RAM, a write sends it "out".
            with cond:
                state['nvme'] = {'in': rd.get(str(NVME_MAJOR), 0), 'out': wr.get(str(NVME_MAJOR), 0),
                                 'procs': rate('@disk_proc')}
                state['zram'] = {'in': rd.get(str(ZRAM_MAJOR), 0), 'out': wr.get(str(ZRAM_MAJOR), 0)}
                state['net'] = {'rx': rate('@net_rx'), 'tx': rate('@net_tx'), 'procs': rate('@net_proc')}
            maps = {}
    # Without bpftrace, most of the picture is gone; exit the whole process
    # (sys.exit in a thread would only end this thread).
    print('bpftrace exited', file=sys.stderr)
    os._exit(1)


# ---------------------------------------------------------------- DRAM via AMD data fabric PMU
#
# eBPF can't see memory-bus traffic: loads and stores don't go through the
# kernel. The memory controllers count it themselves, though. On AMD those
# counters belong to the "data fabric" (the on-chip interconnect between cores,
# iGPU, I/O and the DRAM controllers) and the amd_uncore driver exposes them as
# the amd_df perf PMU. perf_event_open() gives us a file descriptor per
# counter; reading 8 bytes from it returns the running count.

class PerfAttr(ctypes.Structure):
    """The leading fields of struct perf_event_attr; the rest are left at zero (128 bytes total)."""
    _fields_ = [('type', ctypes.c_uint32), ('size', ctypes.c_uint32),
                ('config', ctypes.c_uint64), ('pad', ctypes.c_uint8 * 112)]


def dram_loop():
    subprocess.run(['modprobe', 'amd_uncore'], check=False)
    # Dynamic PMUs get a type number at runtime; perf_event_attr.type must match it.
    pmu_type = read('/sys/bus/event_source/devices/amd_df/type')
    if pmu_type is None:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    fds = []
    # DramChannelDataController0..7: 64-byte beats, reads + writes combined
    # (Zen 2/3 event codes; this chip only populates the first two channels,
    # the rest just count zero). The event number is 12 bits, split across the
    # config word as /sys/bus/event_source/devices/amd_df/format describes:
    # event[7:0] -> config[7:0], event[11:8] -> config[35:32], umask -> config[15:8].
    for ev in (0x07, 0x47, 0x87, 0xC7, 0x107, 0x147, 0x187, 0x1C7):
        a = PerfAttr(type=int(pmu_type), size=128)
        a.config = (ev & 0xff) | (0x38 << 8) | (((ev >> 8) & 0xf) << 32)
        # perf_event_open(attr, pid=-1, cpu=0, group_fd=-1, flags=0). Uncore
        # counters are system-wide, so they're opened as "all processes, on
        # CPU 0"; any CPU would see the same fabric.
        fd = libc.syscall(298, ctypes.byref(a), -1, 0, -1, 0)  # 298 = __NR_perf_event_open on x86_64
        if fd >= 0:
            fds.append(fd)
    peak = dram_peak()
    prev, t0 = None, time.monotonic()
    while True:
        # Counters only go up; the rate is the difference between reads.
        cur = sum(struct.unpack('Q', os.read(fd, 8))[0] for fd in fds)
        t = time.monotonic()
        if prev is not None:
            with cond:
                state['dram'] = {'bw': (cur - prev) * 64 / (t - t0), 'peak': peak}
        prev, t0 = cur, t
        time.sleep(1)


def dram_peak():
    """Theoretical bandwidth from SMBIOS type 17: configured MT/s x data width, summed over devices."""
    # Each "17-N" entry is one memory device's raw SMBIOS record. Parsing it
    # directly avoids depending on dmidecode. Offsets are from the SMBIOS spec:
    #   0x0A  Data Width (bits, WORD)
    #   0x20  Configured Memory Speed (MT/s, WORD; 0xFFFF = see 0x58)
    #   0x58  Extended Configured Memory Speed (MT/s, DWORD, SMBIOS 3.3+)
    total, base = 0, '/sys/firmware/dmi/entries'
    for e in os.listdir(base):
        if not e.startswith('17-'):
            continue
        raw = open(f'{base}/{e}/raw', 'rb').read()
        if len(raw) < 0x22:
            continue
        width = struct.unpack_from('<H', raw, 0x0A)[0]
        speed = struct.unpack_from('<H', raw, 0x20)[0]
        if speed == 0xFFFF and len(raw) >= 0x5C:
            speed = struct.unpack_from('<I', raw, 0x58)[0]
        if width not in (0, 0xFFFF) and speed:     # empty slots report 0 / unknown
            total += speed * 1e6 * width / 8
    return total or None


# ---------------------------------------------------------------- NVIDIA dGPU (NVML), without waking it
#
# Data copied to the dGPU crosses PCIe as DMA, which no kernel tracepoint
# sees, but the GPU counts it. NVML (libnvidia-ml, the library behind
# nvidia-smi) exposes those counters; we call it through ctypes.
#
# The catch on a laptop: the dGPU spends most of its life powered off
# (runtime-suspended, D3cold), and any NVML call wakes it up. So we read the
# kernel's runtime-PM status from sysfs first, which costs nothing, and only
# talk to NVML when the GPU is already awake for some other reason.

class NvMem(ctypes.Structure):
    """nvmlMemory_t"""
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class NvUtil(ctypes.Structure):
    """nvmlUtilization_t: percent of time busy over the last sample period"""
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


def dgpu_loop():
    try:
        nv = ctypes.CDLL('libnvidia-ml.so.1')
    except OSError:
        return
    pm = f'/sys/bus/pci/devices/{DGPU_PCI}/power/runtime_status'  # "active" or "suspended"
    quiet, backoff_until = 0, 0
    while True:
        status = read(pm, 'unknown')
        info = {'status': status}
        # Opening NVML resets the runtime-PM idle timer. If the GPU has been doing nothing,
        # stop asking for a while so it can fall back into D3cold.
        if status == 'active' and time.monotonic() >= backoff_until and nv.nvmlInit_v2() == 0:
            # Init and shut down on every poll rather than holding NVML open:
            # an open handle keeps the device awake.
            h = ctypes.c_void_p()
            nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h))
            tx, rx = ctypes.c_uint(), ctypes.c_uint()
            # Each call samples the PCIe counters for ~20 ms and returns KB/s.
            nv.nvmlDeviceGetPcieThroughput(h, 0, ctypes.byref(tx))  # KB/s, GPU -> host
            nv.nvmlDeviceGetPcieThroughput(h, 1, ctypes.byref(rx))  # KB/s, host -> GPU
            mem, util = NvMem(), NvUtil()
            nv.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem))
            nv.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util))
            nv.nvmlShutdown()
            # GPU -> host lands in DRAM, so it's "in" from DRAM's point of view.
            info.update({'in': tx.value * 1024, 'out': rx.value * 1024, 'vram_used': mem.used,
                         'vram_total': mem.total, 'util': util.gpu})
            # 5 idle polls in a row -> leave it alone for 30 s.
            quiet = quiet + 1 if util.gpu == 0 and tx.value + rx.value < 64 else 0
            if quiet >= 5:
                backoff_until, quiet = time.monotonic() + 30, 0
        elif status == 'active':
            info['status'] = 'idle'   # awake, but we're deliberately not asking
        info['link'] = pcie_link(DGPU_PCI)
        with cond:
            state['dgpu'] = info
        time.sleep(1)


def pcie_link(addr):
    """Current link (it downclocks when idle) and capacity at the fastest rate it can train to."""
    dev = f'/sys/bus/pci/devices/{addr}'
    gt = lambda f: float((read(f'{dev}/{f}', '') or '0').split()[0])   # "16.0 GT/s PCIe" -> 16.0
    # A link runs at the speed of its slower end. The device's parent in sysfs
    # is the CPU root port it's plugged into (e.g. the PCIe 5 GPU on a PCIe 4 port).
    up = os.path.dirname(os.path.realpath(dev))  # the root port on the other end of the link
    gts = gt('current_link_speed')
    max_gts = min(gt('max_link_speed'), float((read(f'{up}/max_link_speed', '') or '999').split()[0]))
    width = int(read(f'{dev}/current_link_width', '0') or 0)
    # Line coding overhead: PCIe 1/2 use 8b/10b, PCIe 3+ use 128b/130b.
    # GT/s is per lane, one bit per transfer, so bytes/s = GT/s x efficiency / 8 x lanes.
    enc = 128 / 130 if max_gts >= 8 else 0.8
    return {'gts': gts, 'width': width, 'cap': max_gts * 1e9 * enc / 8 * width}


# ---------------------------------------------------------------- slow-changing context
#
# Things that label the nodes rather than measure flows: sizes, link speeds,
# device names. Polled every 2 s from procfs/sysfs, which is cheap.

def wifi_bitrate(dev):
    """Negotiated PHY rates from `iw` (needs root). The real throughput limit is well below these."""
    try:
        out = subprocess.run(['iw', 'dev', dev, 'link'], capture_output=True, text=True).stdout
    except OSError:
        return None
    rx = re.search(r'rx bitrate: ([\d.]+) MBit', out)
    tx = re.search(r'tx bitrate: ([\d.]+) MBit', out)
    return {'rx': float(rx.group(1)) * 1e6 / 8 if rx else None,
            'tx': float(tx.group(1)) * 1e6 / 8 if tx else None,
            'associated': out.startswith('Connected to')}


def context_loop():
    while True:
        # /proc/meminfo lines look like "MemTotal:       15584204 kB".
        mi = dict(re.findall(r'(\w+):\s+(\d+)', read('/proc/meminfo', '')))
        kb = lambda k: int(mi.get(k, 0)) * 1024
        # zram mm_stat: orig_data_size compr_data_size mem_used_total ... (bytes)
        mm = (read('/sys/block/zram0/mm_stat', '') or '0 0').split()
        # First line of /proc/stat: cumulative jiffies per CPU state (user nice system idle iowait ...);
        # the page turns successive snapshots into a busy percentage.
        cpu = [int(x) for x in read('/proc/stat').split('\n')[0].split()[1:]]
        # Find the iGPU's DRM card by driver, since card0/card1 numbering isn't stable.
        igpu_card = next((c for c in os.listdir('/sys/class/drm') if re.fullmatch(r'card\d', c)
                          and os.path.basename(os.path.realpath(f'/sys/class/drm/{c}/device/driver')) == 'amdgpu'), None)
        dev = f'/sys/class/drm/{igpu_card}/device' if igpu_card else None
        eth_up = read('/sys/class/net/eno1/carrier') == '1'
        with cond:
            state['mem'] = {'total': kb('MemTotal'), 'avail': kb('MemAvailable'),
                            'cache': kb('Cached'), 'swap_used': kb('SwapTotal') - kb('SwapFree')}
            state['zram_stat'] = {'orig': int(mm[0]), 'compr': int(mm[1])}
            state['cpu_jiffies'] = cpu
            state['cpu_model'] = re.sub(r' with .*', '', re.search(r'model name\s*: (.*)', read('/proc/cpuinfo')).group(1))
            if dev:
                # The iGPU has no memory of its own: "VRAM" is a carve-out of
                # system RAM reserved by firmware, and GTT is ordinary RAM mapped
                # for the GPU on demand. Its traffic is part of the DRAM figure.
                state['igpu'] = {'busy': int(read(f'{dev}/gpu_busy_percent', '0')),
                                 'vram_total': int(read(f'{dev}/mem_info_vram_total', '0')),
                                 'vram_used': int(read(f'{dev}/mem_info_vram_used', '0')),
                                 'gtt_used': int(read(f'{dev}/mem_info_gtt_used', '0'))}
            state['nvme_link'] = pcie_link(NVME_PCI)
            state['nvme_model'] = read('/sys/block/nvme0n1/device/model', 'NVMe')
            state['wifi'] = wifi_bitrate('wlp3s0')
            # /sys/class/net/*/speed is in Mb/s, and errors out when there's no carrier.
            state['eth'] = {'up': eth_up, 'speed': int(read('/sys/class/net/eno1/speed', '0') or 0) * 1e6 / 8 if eth_up else None}
        time.sleep(2)


# ---------------------------------------------------------------- HTTP + SSE
#
# Server-Sent Events: the browser opens /events once and keeps it open; we
# write "data: <json>\n\n" records down it. It's one-way and plain HTTP, and
# EventSource reconnects on its own, so no websocket library is needed.

def publisher():
    """The 1 s clock for the page: wakes every SSE handler to send a snapshot."""
    while True:
        time.sleep(1)
        with cond:
            state['t'] = time.time()
            cond.notify_all()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                  # don't log a line per request

    def do_GET(self):
        if self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            # ThreadingHTTPServer gives each connection its own thread, so it's
            # fine for this one to loop until the browser goes away.
            try:
                while True:
                    with cond:
                        cond.wait()           # until publisher() notifies
                        payload = json.dumps(state)
                    self.wfile.write(f'data: {payload}\n\n'.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        # Anything else gets the page. It's re-read on every request, so edits
        # show up on reload without restarting the server.
        body = open(os.path.join(HERE, 'index.html'), 'rb').read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    # Loading eBPF programs and opening system-wide uncore counters both need
    # root (CAP_BPF / CAP_PERFMON in principle, root in practice).
    if os.geteuid() != 0:
        sys.exit('needs root for eBPF and the DRAM PMU: sudo python3 topod.py')
    # Daemon threads: they die with the main thread on Ctrl+C.
    for fn in (ebpf_loop, dram_loop, dgpu_loop, context_loop, publisher):
        threading.Thread(target=fn, daemon=True).start()
    print(f'http://127.0.0.1:{PORT}', flush=True)
    # Bound to localhost only: the stream includes per-process activity.
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
