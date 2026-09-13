// MUL_MAT_ID micro-bench at the DeepSeek-V4.1-Flash expert shape: 384 experts, [5120 x 2304] MXFP4, top-6.
// The 2.24 GiB expert tensor is allocated through ggml_backend_alloc_ctx_tensors on the CPU backend, i.e. the
// same ggml_backend_cpu_buffer_type path a `-lm none` model load uses, so it also shows whether that path got
// transparent huge pages: after the fill it prints AnonHugePages/Anonymous from /proc/self/smaps_rollup and the
// fill wall time (the fill is where the huge pages are faulted in).
//
// build: g++ -O2 -std=c++17 -I ggml/include benches/thelio-astra/mmid-bench.cpp -o mmid-bench \
//            -L build-cpu/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build-cpu/bin
// run:   [GGML_CPU_NO_THP=1] ./mmid-bench <n_tok> <threads> <iters>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// TLB counters over the timed loop, all threads of this process (perf_event inherit), user space only so it
// works at perf_event_paranoid=2. Raw Arm PMU events: 0x34 DTLB_WALK (page-table walks for data accesses),
// 0x05 L1D_TLB_REFILL, 0x2D L2D_TLB_REFILL. Missing PMU or permission => the counters print as n/a.
struct perf_counter { const char * name; int fd; };
static std::vector<perf_counter> perf_open_tlb() {
    std::vector<perf_counter> v;
#ifdef __linux__
    const struct { const char * name; uint64_t config; } ev[] = { {"DTLB_WALK", 0x34}, {"L1D_TLB_REFILL", 0x05}, {"L2D_TLB_REFILL", 0x2D} };
    for (auto & e : ev) {
        perf_event_attr a; memset(&a, 0, sizeof(a));
        a.type = PERF_TYPE_RAW; a.size = sizeof(a); a.config = e.config;
        a.disabled = 1; a.exclude_kernel = 1; a.exclude_hv = 1; a.inherit = 1;
        int fd = (int) syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
        v.push_back({e.name, fd});
    }
#endif
    return v;
}
static void perf_enable(std::vector<perf_counter> & v, bool on) {
#ifdef __linux__
    for (auto & c : v) if (c.fd >= 0) ioctl(c.fd, on ? PERF_EVENT_IOC_ENABLE : PERF_EVENT_IOC_DISABLE, 0);
#else
    (void) v; (void) on;
#endif
}
static void perf_report(std::vector<perf_counter> & v, int iters) {
    printf("tlb per op:");
    for (auto & c : v) {
        uint64_t n = 0;
        if (c.fd < 0 || read(c.fd, &n, sizeof(n)) != (ssize_t) sizeof(n)) { printf("  %s n/a", c.name); continue; }
        printf("  %s %.0f", c.name, (double) n / iters);
    }
    printf("\n");
}

// system-wide THP/compaction counters, to see what the fill cost the kernel: thp_fault_alloc = huge pages handed
// out at fault time, thp_fault_fallback = faults that wanted one and did not get it, compact_stall = times an
// allocation went into direct compaction (the stall this box has a history of), compact_fail = compaction that
// ran and still failed.
static const char * const vmstat_keys[] = { "thp_fault_alloc", "thp_fault_fallback", "thp_fault_fallback_charge",
                                            "compact_stall", "compact_fail", "compact_success",
                                            "compact_migrate_scanned", "pgmajfault" };
static std::vector<long long> read_vmstat() {
    std::vector<long long> v(sizeof(vmstat_keys)/sizeof(vmstat_keys[0]), -1);
    FILE * f = fopen("/proc/vmstat", "r");
    if (!f) { return v; }
    char key[128]; long long val;
    while (fscanf(f, "%127s %lld", key, &val) == 2) {
        for (size_t i = 0; i < v.size(); i++) { if (!strcmp(key, vmstat_keys[i])) { v[i] = val; } }
    }
    fclose(f);
    return v;
}
static void print_vmstat_delta(const char * when, const std::vector<long long> & a, const std::vector<long long> & b) {
    printf("[%s] vmstat delta:", when);
    for (size_t i = 0; i < b.size(); i++) {
        if (a[i] < 0 || b[i] < 0) { printf("  %s n/a", vmstat_keys[i]); continue; }
        printf("  %s %lld", vmstat_keys[i], b[i] - a[i]);
    }
    printf("\n  (system-wide, so other processes contribute; run on a quiet box)\n");
}

// print the memory counters of this process that tell 4 KiB from 2 MiB backing
static void print_smaps_rollup(const char * when) {
    FILE * f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) { printf("[%s] /proc/self/smaps_rollup unavailable\n", when); return; }
    char line[256];
    std::string rss, anon, anon_huge;
    while (fgets(line, sizeof(line), f)) {
        if      (!strncmp(line, "Rss:",           4))  rss       = line + 4;
        else if (!strncmp(line, "Anonymous:",     10)) anon      = line + 10;
        else if (!strncmp(line, "AnonHugePages:", 14)) anon_huge = line + 14;
    }
    fclose(f);
    auto trim = [](std::string s) { size_t a = s.find_first_not_of(" \t"); size_t b = s.find_last_not_of(" \t\n"); return a == std::string::npos ? std::string() : s.substr(a, b - a + 1); };
    printf("[%s] smaps_rollup: Rss %s | Anonymous %s | AnonHugePages %s\n", when, trim(rss).c_str(), trim(anon).c_str(), trim(anon_huge).c_str());
}

int main(int argc, char ** argv) {
    const int64_t n_embd = 5120, n_ff = 2304, n_exp = 384, n_used = 6;
    int n_tok = argc > 1 ? atoi(argv[1]) : 1;
    int nth   = argc > 2 ? atoi(argv[2]) : 64;
    int iters = argc > 3 ? atoi(argv[3]) : 100;
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    std::vector<perf_counter> tlb = perf_open_tlb(); // before any compute thread exists, so all of them inherit the counters
    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, n_embd, n_ff, n_exp);  // gate/up shape
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    printf("weights: %.2f GiB MXFP4 in a %s buffer of %.2f GiB\n", ggml_nbytes(w)/1073741824.0,
        ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1073741824.0);
    print_smaps_rollup("after alloc, before fill");
    // fill weights: e8m0 exponent = 127 (scale 1.0), random nibbles
    double fill_s = 0;
    {
        const size_t nb = ggml_nbytes(w); std::vector<uint8_t> chunk(17*4096);
        uint64_t s = 0x9E3779B97F4A7C15ull;
        std::vector<long long> vm0 = read_vmstat();
        auto f0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < nb; off += chunk.size()) {
            size_t n = std::min(chunk.size(), nb - off);
            for (size_t i = 0; i < n; i += 17) { chunk[i] = 127; for (int j = 1; j < 17 && i+j < n; j++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; chunk[i+j] = (uint8_t)s; } }
            ggml_backend_tensor_set(w, chunk.data(), off, n);
        }
        auto f1 = std::chrono::steady_clock::now();
        fill_s = std::chrono::duration<double>(f1-f0).count();
        printf("fill: %.3f s for %.2f GiB (%.2f GB/s, single thread, includes generating the random bytes)\n", fill_s, nb/1073741824.0, nb/fill_s/1e9);
        print_vmstat_delta("fill", vm0, read_vmstat());
        std::vector<float> xf(n_embd*n_used*n_tok); for (auto & v : xf) v = (rand()/(float)RAND_MAX) - 0.5f;
        ggml_backend_tensor_set(x, xf.data(), 0, xf.size()*4);
    }
    print_smaps_rollup("after fill");
    ggml_init_params gp = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * out = ggml_mul_mat_id(gctx, w, x, ids);
    ggml_cgraph * gf = ggml_new_graph(gctx); ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    std::mt19937 rng(42); std::vector<int32_t> idv(n_used*n_tok);
    auto set_ids = [&](){ for (int t = 0; t < n_tok; t++) { std::vector<int> p(n_exp); for (int i=0;i<n_exp;i++) p[i]=i; std::shuffle(p.begin(), p.end(), rng); for (int u=0;u<n_used;u++) idv[t*n_used+u]=p[u]; } ggml_backend_tensor_set(ids, idv.data(), 0, idv.size()*4); };
    for (int i = 0; i < 3; i++) { set_ids(); ggml_backend_graph_compute(be, gf); }
    double tot = 0; size_t bytes_tot = 0;
    perf_enable(tlb, true);
    for (int i = 0; i < iters; i++) {
        set_ids();
        std::vector<char> used(n_exp, 0); for (auto v : idv) used[v] = 1; size_t nd = std::count(used.begin(), used.end(), 1);
        auto t0 = std::chrono::steady_clock::now(); ggml_backend_graph_compute(be, gf); auto t1 = std::chrono::steady_clock::now();
        tot += std::chrono::duration<double>(t1-t0).count(); bytes_tot += nd * (size_t)w->nb[2];
    }
    perf_enable(tlb, false);
    double ms = tot/iters*1e3; double gbs = bytes_tot/tot/1e9; double gflops = 2.0*n_embd*n_ff*n_used*n_tok*iters/tot/1e9;
    printf("n_tok=%d threads=%d : %.3f ms/op  %.1f GB/s of expert bytes (%.1f MB/op)  %.0f GFLOP/s\n", n_tok, nth, ms, gbs, (double)bytes_tot/iters/1e6, gflops);
    perf_report(tlb, iters);
    return 0;
}
