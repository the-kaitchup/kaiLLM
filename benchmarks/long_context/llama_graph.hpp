// Optional Linux benchmark instrumentation for the three pinned llama.cpp references.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kaillm_graph {
enum phase : uint64_t {
    unattributed, prefill, decode, mixed, proposal, draft_prefill, draft_decode, draft_mixed, phase_count
};
enum event : uint64_t {
    batches, captures, capture_launches, replays, rejected, updates, proposals, completed_proposals, event_count
};
constexpr size_t devices = 4;
constexpr uint64_t magic = UINT64_C(0x4b41494752415048);
struct counters {
    uint64_t signature, version, device_count, active_phase;
    uint64_t value[phase_count][devices][event_count];
};
inline void fail(const char * why) {
    std::fprintf(stderr, "llama.cpp graph proof: %s\n", why);
    std::abort();
}
inline counters * open_counters() {
    const char * name = std::getenv("KAILLM_LLAMA_GRAPH_FILE");
    if (!name) return nullptr;
    const int fd = ::open(name, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    struct stat info {};
    if (fd < 0 || ::fstat(fd, &info) || !S_ISREG(info.st_mode) ||
            info.st_uid != ::geteuid() || (info.st_mode & 077) || info.st_size != static_cast<off_t>(sizeof(counters))) {
        fail("invalid counters file");
    }
    void * ptr = ::mmap(nullptr, sizeof(counters), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) fail("cannot map counters");
    auto * data = static_cast<counters *>(ptr);
    if (data->signature != magic || data->version != 2 || (data->device_count != 1 && data->device_count != 2 &&
            data->device_count != devices)) fail("wrong counters layout");
    return data;
}
inline counters * data() {
    static counters * const shared = open_counters();
    return shared;
}
inline void record(int device, event item) {
    auto * shared = data();
    if (shared == nullptr) return;
    if (device < 0 || static_cast<uint64_t>(device) >= shared->device_count) fail("untracked CUDA device");
    const uint64_t current = __atomic_load_n(&shared->active_phase, __ATOMIC_ACQUIRE);
    if (current >= phase_count) fail("invalid phase");
    __atomic_fetch_add(&shared->value[current][device][item], UINT64_C(1), __ATOMIC_RELAXED);
}
class scope {
    counters * shared;
    const phase current;
    uint64_t before[devices] = {};
    uint64_t launches(uint64_t device) const {
        const auto & events = shared->value[proposal][device];
        return __atomic_load_n(&events[capture_launches], __ATOMIC_ACQUIRE) +
               __atomic_load_n(&events[replays], __ATOMIC_ACQUIRE);
    }
public:
    explicit scope(phase id) : shared(data()), current(id) {
        if (shared == nullptr) return;
        const uint64_t previous = __atomic_exchange_n(&shared->active_phase, static_cast<uint64_t>(current), __ATOMIC_ACQ_REL);
        if (previous != unattributed) fail("overlapping model phases");
        if (current == proposal) {
            for (uint64_t i = 0; i < shared->device_count; ++i) {
                before[i] = launches(i);
                record(i, proposals);
            }
        }
    }
    scope(const scope &) = delete;
    ~scope() {
        if (shared == nullptr) return;
        if (current == proposal) {
            for (uint64_t i = 0; i < shared->device_count; ++i) {
                if (launches(i) > before[i]) record(i, completed_proposals);
            }
        }
        __atomic_store_n(&shared->active_phase, uint64_t(unattributed), __ATOMIC_RELEASE);
    }
};
} // namespace kaillm_graph
