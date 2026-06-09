#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include <mpi.h>
#include <omp.h>

#ifdef _WIN32
#include <direct.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using ResultHeap = std::priority_queue<std::pair<float, uint32_t> >;

struct Config {
    std::string algo;
    std::string data_path;
    std::string partition;
    std::string group;
    size_t nlist;
    size_t nprobe;
    size_t max_queries;
    size_t k;
    size_t train_n;
    int iters;
    int threads;
    int nodes;
    int ppn;
    std::string run_id;
    bool rebuild_centroids;

    Config()
        : algo("ivf-mpi-3-3"),
          data_path("/anndata/"),
          partition("block-id"),
          group("manual"),
          nlist(512),
          nprobe(64),
          max_queries(2000),
          k(10),
          train_n(12000),
          iters(6),
          threads(1),
          nodes(0),
          ppn(0),
          run_id("manual"),
          rebuild_centroids(false) {}
};

struct LocalIVFIndex {
    size_t nlist;
    size_t vecdim;
    size_t global_begin;
    size_t global_end;
    std::vector<uint32_t> list_offsets;
    std::vector<uint32_t> ids;
    std::vector<float> vectors;
};

struct QueryMetrics {
    double recall_sum;
    double latency_us_sum;
    double bcast_us_sum;
    double local_search_avg_us_sum;
    double local_search_max_us_sum;
    double gather_us_sum;
    double merge_us_sum;
    double candidate_count_sum;

    QueryMetrics()
        : recall_sum(0.0),
          latency_us_sum(0.0),
          bcast_us_sum(0.0),
          local_search_avg_us_sum(0.0),
          local_search_max_us_sum(0.0),
          gather_us_sum(0.0),
          merge_us_sum(0.0),
          candidate_count_sum(0.0) {}
};

static bool starts_with(const std::string &s, const char *prefix) {
    const size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

static std::string value_after_equal(const std::string &arg) {
    const size_t pos = arg.find('=');
    return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
}

static size_t parse_size_value(const std::string &s, size_t fallback) {
    if (s.empty()) {
        return fallback;
    }
    const long long parsed = std::atoll(s.c_str());
    return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
}

static int parse_int_value(const std::string &s, int fallback) {
    if (s.empty()) {
        return fallback;
    }
    const int parsed = std::atoi(s.c_str());
    return parsed > 0 ? parsed : fallback;
}

static void print_usage(int rank) {
    if (rank != 0) {
        return;
    }

    std::cout
        << "Usage:\n"
        << "  mpiexec -np <np> ./main --algo=ivf-mpi-3-3 --nlist=512 --nprobe=64 --test=2000 --threads=2 --k=10\n"
        << "\n"
        << "Options:\n"
        << "  --algo=ivf-mpi-3-3      Algorithm label printed in results\n"
        << "  --data=/anndata/        Dataset directory\n"
        << "  --nlist=512             IVF list count\n"
        << "  --nprobe=64             Number of probed IVF lists per query\n"
        << "  --test=2000             Max query count\n"
        << "  --k=10                  Recall@k and returned top-k\n"
        << "  --train=12000           Training sample count for centroids\n"
        << "  --iters=6               K-means iteration count\n"
        << "  --partition=block-id    block-id, contiguous-list, or greedy-list\n"
        << "  --group=manual          Experiment group label recorded in CSV\n"
        << "  --nodes=<nodes>         Optional record-only qsub nodes value\n"
        << "  --ppn=<ppn>             Optional record-only qsub ppn value\n"
        << "  --threads=1             OpenMP threads per MPI rank\n"
        << "  --run-id=manual         Unique run id recorded in CSV\n"
        << "  --rebuild-centroids     Ignore centroid cache and retrain\n"
        << "\n"
        << "Backward-compatible positional form:\n"
        << "  ./main <nlist> <nprobe> <max_queries>\n";
}

static Config parse_args(int argc, char **argv, int rank) {
    Config cfg;
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(rank);
            MPI_Finalize();
            std::exit(0);
        } else if (starts_with(arg, "--algo=")) {
            cfg.algo = value_after_equal(arg);
        } else if (starts_with(arg, "--data=")) {
            cfg.data_path = value_after_equal(arg);
            if (!cfg.data_path.empty() && cfg.data_path[cfg.data_path.size() - 1] != '/') {
                cfg.data_path += "/";
            }
        } else if (starts_with(arg, "--nlist=")) {
            cfg.nlist = parse_size_value(value_after_equal(arg), cfg.nlist);
        } else if (starts_with(arg, "--nprobe=")) {
            cfg.nprobe = parse_size_value(value_after_equal(arg), cfg.nprobe);
        } else if (starts_with(arg, "--test=")) {
            cfg.max_queries = parse_size_value(value_after_equal(arg), cfg.max_queries);
        } else if (starts_with(arg, "--k=")) {
            cfg.k = parse_size_value(value_after_equal(arg), cfg.k);
        } else if (starts_with(arg, "--train=")) {
            cfg.train_n = parse_size_value(value_after_equal(arg), cfg.train_n);
        } else if (starts_with(arg, "--iters=")) {
            cfg.iters = parse_int_value(value_after_equal(arg), cfg.iters);
        } else if (starts_with(arg, "--partition=")) {
            cfg.partition = value_after_equal(arg);
        } else if (starts_with(arg, "--group=")) {
            cfg.group = value_after_equal(arg);
        } else if (starts_with(arg, "--threads=")) {
            cfg.threads = parse_int_value(value_after_equal(arg), cfg.threads);
        } else if (starts_with(arg, "--nodes=")) {
            cfg.nodes = parse_int_value(value_after_equal(arg), cfg.nodes);
        } else if (starts_with(arg, "--ppn=")) {
            cfg.ppn = parse_int_value(value_after_equal(arg), cfg.ppn);
        } else if (starts_with(arg, "--run-id=")) {
            cfg.run_id = value_after_equal(arg);
        } else if (arg == "--rebuild-centroids") {
            cfg.rebuild_centroids = true;
        } else if (!arg.empty() && arg[0] != '-') {
            if (positional == 0) {
                cfg.nlist = parse_size_value(arg, cfg.nlist);
            } else if (positional == 1) {
                cfg.nprobe = parse_size_value(arg, cfg.nprobe);
            } else if (positional == 2) {
                cfg.max_queries = parse_size_value(arg, cfg.max_queries);
            }
            ++positional;
        }
    }

    cfg.threads = std::max(1, std::min(cfg.threads, 64));
    if (cfg.partition == "block" || cfg.partition == "block-by-id") {
        cfg.partition = "block-id";
    } else if (cfg.partition == "contiguous" || cfg.partition == "list" ||
               cfg.partition == "contiguous-lists") {
        cfg.partition = "contiguous-list";
    } else if (cfg.partition == "greedy" || cfg.partition == "balanced" ||
               cfg.partition == "greedy-balanced" || cfg.partition == "greedy-lists") {
        cfg.partition = "greedy-list";
    } else if (cfg.partition != "block-id" &&
               cfg.partition != "contiguous-list" &&
               cfg.partition != "greedy-list") {
        if (rank == 0) {
            std::cerr << "warning: unknown --partition value, using block-id\n";
        }
        cfg.partition = "block-id";
    }
    if (cfg.group.empty()) {
        cfg.group = "manual";
    }
    cfg.iters = std::max(1, cfg.iters);
    cfg.k = std::max<size_t>(1, cfg.k);
    cfg.nprobe = std::max<size_t>(1, cfg.nprobe);
    cfg.nlist = std::max<size_t>(1, cfg.nlist);
    cfg.max_queries = std::max<size_t>(1, cfg.max_queries);
    return cfg;
}

static void abort_all(const std::string &message, int rank) {
    std::cerr << "rank " << rank << " error: " << message << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

template <typename T>
static T *LoadDataOrAbort(const std::string &data_path, size_t &n, size_t &d, int rank) {
    std::ifstream fin(data_path.c_str(), std::ios::in | std::ios::binary);
    if (!fin) {
        abort_all("failed to open " + data_path, rank);
    }

    uint32_t n32 = 0;
    uint32_t d32 = 0;
    fin.read(reinterpret_cast<char *>(&n32), sizeof(uint32_t));
    fin.read(reinterpret_cast<char *>(&d32), sizeof(uint32_t));
    if (!fin) {
        abort_all("failed to read header from " + data_path, rank);
    }

    n = static_cast<size_t>(n32);
    d = static_cast<size_t>(d32);
    T *data = new T[n * d];
    fin.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(n * d * sizeof(T)));
    if (!fin) {
        delete[] data;
        abort_all("failed to read full file " + data_path, rank);
    }

    std::cerr << "rank " << rank << " load data " << data_path
              << " dimension=" << d << " number=" << n
              << " size_per_element=" << sizeof(T) << "\n";
    return data;
}

static int mpi_count(size_t value, const char *name, int rank) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abort_all(std::string("MPI count too large for ") + name, rank);
    }
    return static_cast<int>(value);
}

static void ensure_files_dir() {
#ifdef _WIN32
    _mkdir("files");
#else
    mkdir("files", 0755);
#endif
}

static bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

static std::string cache_token(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') ||
                        ch == '-';
        out.push_back(ok ? ch : '-');
    }
    return out.empty() ? "none" : out;
}

static std::string centroid_cache_path(size_t nlist, size_t vecdim, size_t base_number) {
    std::ostringstream oss;
    oss << "files/ivf_centroids_nlist" << nlist
        << "_d" << vecdim
        << "_n" << base_number << ".bin";
    return oss.str();
}

static std::string local_ivf_cache_path(size_t nlist,
                                        size_t vecdim,
                                        size_t base_number,
                                        int world_size,
                                        int rank,
                                        const std::string &partition,
                                        size_t begin,
                                        size_t end) {
    std::ostringstream oss;
    oss << "files/ivf_mpi_local_nlist" << nlist
        << "_d" << vecdim
        << "_n" << base_number
        << "_np" << world_size
        << "_rank" << rank
        << "_part" << cache_token(partition)
        << "_b" << begin
        << "_e" << end << ".bin";
    return oss.str();
}

template <typename T>
static void write_pod_vector(std::ofstream &out, const std::vector<T> &v) {
    const uint64_t size = static_cast<uint64_t>(v.size());
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    if (!v.empty()) {
        out.write(reinterpret_cast<const char *>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(T)));
    }
}

template <typename T>
static bool read_pod_vector(std::ifstream &in, std::vector<T> &v) {
    uint64_t size = 0;
    in.read(reinterpret_cast<char *>(&size), sizeof(size));
    if (!in) {
        return false;
    }
    v.assign(static_cast<size_t>(size), T());
    if (!v.empty()) {
        in.read(reinterpret_cast<char *>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(T)));
    }
    return static_cast<bool>(in);
}

static bool load_centroids(std::vector<float> &centroids,
                           const std::string &path,
                           size_t expected_nlist,
                           size_t expected_vecdim,
                           size_t expected_base_number) {
    std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[8];
    uint64_t nlist = 0;
    uint64_t vecdim = 0;
    uint64_t base_number = 0;
    uint64_t count = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&nlist), sizeof(nlist));
    in.read(reinterpret_cast<char *>(&vecdim), sizeof(vecdim));
    in.read(reinterpret_cast<char *>(&base_number), sizeof(base_number));
    in.read(reinterpret_cast<char *>(&count), sizeof(count));

    const char expected_magic[8] = {'I', 'V', 'F', 'C', 'E', 'N', 'T', '1'};
    if (!in || std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        nlist != expected_nlist || vecdim != expected_vecdim ||
        base_number != expected_base_number ||
        count != expected_nlist * expected_vecdim) {
        return false;
    }

    centroids.assign(static_cast<size_t>(count), 0.0f);
    in.read(reinterpret_cast<char *>(centroids.data()),
            static_cast<std::streamsize>(centroids.size() * sizeof(float)));
    return static_cast<bool>(in);
}

static bool save_centroids(const std::vector<float> &centroids,
                           const std::string &path,
                           size_t nlist,
                           size_t vecdim,
                           size_t base_number) {
    ensure_files_dir();
    std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);
    if (!out) {
        return false;
    }

    const char magic[8] = {'I', 'V', 'F', 'C', 'E', 'N', 'T', '1'};
    const uint64_t nlist64 = static_cast<uint64_t>(nlist);
    const uint64_t vecdim64 = static_cast<uint64_t>(vecdim);
    const uint64_t base64 = static_cast<uint64_t>(base_number);
    const uint64_t count64 = static_cast<uint64_t>(centroids.size());
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&nlist64), sizeof(nlist64));
    out.write(reinterpret_cast<const char *>(&vecdim64), sizeof(vecdim64));
    out.write(reinterpret_cast<const char *>(&base64), sizeof(base64));
    out.write(reinterpret_cast<const char *>(&count64), sizeof(count64));
    out.write(reinterpret_cast<const char *>(centroids.data()),
              static_cast<std::streamsize>(centroids.size() * sizeof(float)));
    return static_cast<bool>(out);
}

#ifdef __ARM_NEON
static inline float hsum_f32x4(float32x4_t v) {
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}
#endif

static inline float inner_product_neon(const float *a, const float *b, size_t dim) {
#ifdef __ARM_NEON
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    size_t d = 0;
    for (; d + 15 < dim; d += 16) {
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + d), vld1q_f32(b + d));
        sum1 = vmlaq_f32(sum1, vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
        sum2 = vmlaq_f32(sum2, vld1q_f32(a + d + 8), vld1q_f32(b + d + 8));
        sum3 = vmlaq_f32(sum3, vld1q_f32(a + d + 12), vld1q_f32(b + d + 12));
    }

    float result = hsum_f32x4(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (; d < dim; ++d) {
        result += a[d] * b[d];
    }
    return result;
#else
    float result = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        result += a[d] * b[d];
    }
    return result;
#endif
}

static inline float inner_product_neon_96(const float *a, const float *b) {
#ifdef __ARM_NEON
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (int d = 0; d < 96; d += 16) {
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + d), vld1q_f32(b + d));
        sum1 = vmlaq_f32(sum1, vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
        sum2 = vmlaq_f32(sum2, vld1q_f32(a + d + 8), vld1q_f32(b + d + 8));
        sum3 = vmlaq_f32(sum3, vld1q_f32(a + d + 12), vld1q_f32(b + d + 12));
    }

    return hsum_f32x4(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
#else
    float result = 0.0f;
    for (int d = 0; d < 96; ++d) {
        result += a[d] * b[d];
    }
    return result;
#endif
}

static inline float inner_product_auto(const float *a, const float *b, size_t dim) {
    return dim == 96 ? inner_product_neon_96(a, b) : inner_product_neon(a, b, dim);
}

static inline void push_topk(ResultHeap &heap, float dis, uint32_t id, size_t k) {
    if (heap.size() < k) {
        heap.push(std::make_pair(dis, id));
    } else if (dis < heap.top().first) {
        heap.pop();
        heap.push(std::make_pair(dis, id));
    }
}

static ResultHeap merge_heaps(std::vector<ResultHeap> &heaps, size_t k) {
    ResultHeap merged;
    for (size_t i = 0; i < heaps.size(); ++i) {
        while (!heaps[i].empty()) {
            const std::pair<float, uint32_t> item = heaps[i].top();
            heaps[i].pop();
            push_topk(merged, item.first, item.second, k);
        }
    }
    return merged;
}

static std::vector<uint32_t> heap_to_ids(ResultHeap heap) {
    std::vector<uint32_t> ids;
    ids.reserve(heap.size());
    while (!heap.empty()) {
        ids.push_back(heap.top().second);
        heap.pop();
    }
    return ids;
}

static std::vector<std::pair<float, uint32_t> > heap_to_sorted_items(ResultHeap heap, size_t k) {
    std::vector<std::pair<float, uint32_t> > items;
    items.reserve(k);
    while (!heap.empty()) {
        items.push_back(heap.top());
        heap.pop();
    }
    std::sort(items.begin(), items.end());
    return items;
}

static uint32_t find_nearest_centroid_ip(const float *vec,
                                         const std::vector<float> &centroids,
                                         size_t nlist,
                                         size_t vecdim) {
    float best_dis = std::numeric_limits<float>::max();
    uint32_t best_id = 0;
    for (size_t c = 0; c < nlist; ++c) {
        const float *centroid = centroids.data() + c * vecdim;
        const float dis = 1.0f - inner_product_auto(centroid, vec, vecdim);
        if (dis < best_dis) {
            best_dis = dis;
            best_id = static_cast<uint32_t>(c);
        }
    }
    return best_id;
}

static void train_ivf_centroids(const float *base,
                                size_t base_number,
                                size_t vecdim,
                                size_t nlist,
                                size_t train_n,
                                int iters,
                                std::vector<float> &centroids) {
    train_n = std::max<size_t>(1, std::min(train_n, base_number));
    const size_t stride = std::max<size_t>(1, base_number / train_n);

    centroids.assign(nlist * vecdim, 0.0f);
    for (size_t c = 0; c < nlist; ++c) {
        const size_t id = (c * stride * 7 + c * 13) % base_number;
        std::memcpy(centroids.data() + c * vecdim,
                    base + id * vecdim,
                    sizeof(float) * vecdim);
    }

    std::vector<float> sums(nlist * vecdim);
    std::vector<uint32_t> counts(nlist);

    for (int it = 0; it < iters; ++it) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t t = 0; t < train_n; ++t) {
            const size_t id = (t * stride) % base_number;
            const float *vec = base + id * vecdim;
            const uint32_t cid = find_nearest_centroid_ip(vec, centroids, nlist, vecdim);
            ++counts[cid];

            float *sum = sums.data() + static_cast<size_t>(cid) * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                sum[d] += vec[d];
            }
        }

        size_t empty_count = 0;
        for (size_t c = 0; c < nlist; ++c) {
            float *centroid = centroids.data() + c * vecdim;
            if (counts[c] == 0) {
                const size_t id = ((c + 1) * 7919 + static_cast<size_t>(it + 1) * 104729) % base_number;
                std::memcpy(centroid, base + id * vecdim, sizeof(float) * vecdim);
                ++empty_count;
                continue;
            }

            const float inv = 1.0f / static_cast<float>(counts[c]);
            const float *sum = sums.data() + c * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                centroid[d] = sum[d] * inv;
            }
        }

        std::cerr << "ivf kmeans iter " << (it + 1) << "/" << iters
                  << ", empty clusters: " << empty_count << "\n";
    }
}

static void partition_block(size_t total, int rank, int size, size_t &begin, size_t &end) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    begin = total * r / p;
    end = total * (r + 1) / p;
}

static std::vector<int> contiguous_list_owners(size_t nlist, int world_size) {
    std::vector<int> owners(nlist, 0);
    for (size_t c = 0; c < nlist; ++c) {
        owners[c] = static_cast<int>(c * static_cast<size_t>(world_size) / nlist);
        if (owners[c] >= world_size) {
            owners[c] = world_size - 1;
        }
    }
    return owners;
}

static std::vector<int> greedy_list_owners(const std::vector<uint32_t> &counts, int world_size) {
    std::vector<size_t> list_ids(counts.size(), 0);
    for (size_t c = 0; c < counts.size(); ++c) {
        list_ids[c] = c;
    }
    std::sort(list_ids.begin(), list_ids.end(),
              [&counts](size_t a, size_t b) {
                  if (counts[a] != counts[b]) {
                      return counts[a] > counts[b];
                  }
                  return a < b;
              });

    std::vector<unsigned long long> load(static_cast<size_t>(world_size), 0);
    std::vector<int> owners(counts.size(), 0);
    for (size_t i = 0; i < list_ids.size(); ++i) {
        int best_rank = 0;
        for (int r = 1; r < world_size; ++r) {
            if (load[static_cast<size_t>(r)] < load[static_cast<size_t>(best_rank)]) {
                best_rank = r;
            }
        }
        const size_t list_id = list_ids[i];
        owners[list_id] = best_rank;
        load[static_cast<size_t>(best_rank)] += counts[list_id];
    }
    return owners;
}

static LocalIVFIndex build_local_ivf_index(const float *base,
                                           size_t base_number,
                                           size_t vecdim,
                                           const std::vector<float> &centroids,
                                           size_t nlist,
                                           int rank,
                                           int world_size,
                                           const std::string &partition) {
    size_t begin = 0;
    size_t end = 0;
    if (partition == "block-id") {
        partition_block(base_number, rank, world_size, begin, end);
    } else {
        begin = 0;
        end = base_number;
    }

    LocalIVFIndex index;
    index.nlist = nlist;
    index.vecdim = vecdim;
    index.global_begin = begin;
    index.global_end = end;

    std::vector<uint32_t> counts(nlist, 0);

    if (partition == "block-id") {
        const size_t local_count = end - begin;
        std::vector<uint32_t> assignments(local_count);

        for (size_t local = 0; local < local_count; ++local) {
            const size_t global_id = begin + local;
            const uint32_t cid = find_nearest_centroid_ip(
                base + global_id * vecdim, centroids, nlist, vecdim);
            assignments[local] = cid;
            ++counts[cid];
        }

        index.list_offsets.assign(nlist + 1, 0);
        for (size_t c = 0; c < nlist; ++c) {
            index.list_offsets[c + 1] = index.list_offsets[c] + counts[c];
        }

        index.ids.assign(local_count, 0);
        index.vectors.assign(local_count * vecdim, 0.0f);
        std::vector<uint32_t> cursor = index.list_offsets;

        for (size_t local = 0; local < local_count; ++local) {
            const size_t global_id = begin + local;
            const uint32_t cid = assignments[local];
            const uint32_t pos = cursor[cid]++;
            index.ids[pos] = static_cast<uint32_t>(global_id);
            std::memcpy(index.vectors.data() + static_cast<size_t>(pos) * vecdim,
                        base + global_id * vecdim,
                        sizeof(float) * vecdim);
        }

        return index;
    }

    std::vector<uint32_t> assignments(base_number);
    for (size_t global_id = 0; global_id < base_number; ++global_id) {
        const uint32_t cid = find_nearest_centroid_ip(
            base + global_id * vecdim, centroids, nlist, vecdim);
        assignments[global_id] = cid;
        ++counts[cid];
    }

    const std::vector<int> owners = partition == "greedy-list"
        ? greedy_list_owners(counts, world_size)
        : contiguous_list_owners(nlist, world_size);

    index.list_offsets.assign(nlist + 1, 0);
    for (size_t c = 0; c < nlist; ++c) {
        const uint32_t owned_count = owners[c] == rank ? counts[c] : 0;
        index.list_offsets[c + 1] = index.list_offsets[c] + owned_count;
    }

    const size_t local_count = index.list_offsets.back();
    index.ids.assign(local_count, 0);
    index.vectors.assign(local_count * vecdim, 0.0f);
    std::vector<uint32_t> cursor = index.list_offsets;

    for (size_t global_id = 0; global_id < base_number; ++global_id) {
        const uint32_t cid = assignments[global_id];
        if (owners[cid] != rank) {
            continue;
        }
        const uint32_t pos = cursor[cid]++;
        index.ids[pos] = static_cast<uint32_t>(global_id);
        std::memcpy(index.vectors.data() + static_cast<size_t>(pos) * vecdim,
                    base + global_id * vecdim,
                    sizeof(float) * vecdim);
    }

    return index;
}

static bool save_local_ivf_index(const LocalIVFIndex &index,
                                 const std::string &path,
                                 size_t base_number,
                                 int world_size,
                                 int rank) {
    ensure_files_dir();
    std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);
    if (!out) {
        return false;
    }

    const char magic[8] = {'I', 'V', 'F', 'M', 'L', 'O', 'C', '1'};
    const uint64_t nlist = static_cast<uint64_t>(index.nlist);
    const uint64_t vecdim = static_cast<uint64_t>(index.vecdim);
    const uint64_t base = static_cast<uint64_t>(base_number);
    const uint64_t np = static_cast<uint64_t>(world_size);
    const uint64_t r = static_cast<uint64_t>(rank);
    const uint64_t begin = static_cast<uint64_t>(index.global_begin);
    const uint64_t end = static_cast<uint64_t>(index.global_end);

    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&nlist), sizeof(nlist));
    out.write(reinterpret_cast<const char *>(&vecdim), sizeof(vecdim));
    out.write(reinterpret_cast<const char *>(&base), sizeof(base));
    out.write(reinterpret_cast<const char *>(&np), sizeof(np));
    out.write(reinterpret_cast<const char *>(&r), sizeof(r));
    out.write(reinterpret_cast<const char *>(&begin), sizeof(begin));
    out.write(reinterpret_cast<const char *>(&end), sizeof(end));
    write_pod_vector(out, index.list_offsets);
    write_pod_vector(out, index.ids);
    write_pod_vector(out, index.vectors);
    return static_cast<bool>(out);
}

static bool load_local_ivf_index(LocalIVFIndex &index,
                                 const std::string &path,
                                 size_t expected_nlist,
                                 size_t expected_vecdim,
                                 size_t expected_base_number,
                                 int expected_world_size,
                                 int expected_rank,
                                 size_t expected_begin,
                                 size_t expected_end) {
    std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[8];
    uint64_t nlist = 0;
    uint64_t vecdim = 0;
    uint64_t base = 0;
    uint64_t np = 0;
    uint64_t r = 0;
    uint64_t begin = 0;
    uint64_t end = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&nlist), sizeof(nlist));
    in.read(reinterpret_cast<char *>(&vecdim), sizeof(vecdim));
    in.read(reinterpret_cast<char *>(&base), sizeof(base));
    in.read(reinterpret_cast<char *>(&np), sizeof(np));
    in.read(reinterpret_cast<char *>(&r), sizeof(r));
    in.read(reinterpret_cast<char *>(&begin), sizeof(begin));
    in.read(reinterpret_cast<char *>(&end), sizeof(end));

    const char expected_magic[8] = {'I', 'V', 'F', 'M', 'L', 'O', 'C', '1'};
    if (!in || std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        nlist != expected_nlist ||
        vecdim != expected_vecdim ||
        base != expected_base_number ||
        np != static_cast<uint64_t>(expected_world_size) ||
        r != static_cast<uint64_t>(expected_rank) ||
        begin != expected_begin ||
        end != expected_end) {
        return false;
    }

    LocalIVFIndex loaded;
    loaded.nlist = static_cast<size_t>(nlist);
    loaded.vecdim = static_cast<size_t>(vecdim);
    loaded.global_begin = static_cast<size_t>(begin);
    loaded.global_end = static_cast<size_t>(end);
    if (!read_pod_vector(in, loaded.list_offsets) ||
        !read_pod_vector(in, loaded.ids) ||
        !read_pod_vector(in, loaded.vectors)) {
        return false;
    }

    if (loaded.list_offsets.size() != loaded.nlist + 1 ||
        loaded.vectors.size() != loaded.ids.size() * loaded.vecdim ||
        loaded.list_offsets.empty() ||
        loaded.list_offsets.front() != 0 ||
        loaded.list_offsets.back() != loaded.ids.size()) {
        return false;
    }

    index = std::move(loaded);
    return true;
}

static std::vector<uint32_t> select_ivf_lists(const std::vector<float> &centroids,
                                              size_t nlist,
                                              size_t vecdim,
                                              const float *query,
                                              size_t nprobe) {
    nprobe = std::max<size_t>(1, std::min(nprobe, nlist));
    ResultHeap heap;
    for (size_t c = 0; c < nlist; ++c) {
        const float *centroid = centroids.data() + c * vecdim;
        const float dis = 1.0f - inner_product_auto(centroid, query, vecdim);
        push_topk(heap, dis, static_cast<uint32_t>(c), nprobe);
    }
    return heap_to_ids(heap);
}

static std::vector<uint32_t> select_ivf_lists_omp(const std::vector<float> &centroids,
                                                  size_t nlist,
                                                  size_t vecdim,
                                                  const float *query,
                                                  size_t nprobe,
                                                  int threads) {
    if (threads <= 1) {
        return select_ivf_lists(centroids, nlist, vecdim, query, nprobe);
    }

    nprobe = std::max<size_t>(1, std::min(nprobe, nlist));
    std::vector<ResultHeap> local_heaps(static_cast<size_t>(threads));

#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        ResultHeap local;

#pragma omp for schedule(static)
        for (int64_t c = 0; c < static_cast<int64_t>(nlist); ++c) {
            const float *centroid = centroids.data() + static_cast<size_t>(c) * vecdim;
            const float dis = 1.0f - inner_product_auto(centroid, query, vecdim);
            push_topk(local, dis, static_cast<uint32_t>(c), nprobe);
        }

        local_heaps[static_cast<size_t>(tid)] = std::move(local);
    }

    ResultHeap merged = merge_heaps(local_heaps, nprobe);
    return heap_to_ids(merged);
}

static ResultHeap search_local_ivf(const LocalIVFIndex &index,
                                   const float *query,
                                   const std::vector<uint32_t> &selected_lists,
                                   size_t k,
                                   uint64_t &candidate_count) {
    candidate_count = 0;
    ResultHeap result;

    for (size_t p = 0; p < selected_lists.size(); ++p) {
        const uint32_t list_id = selected_lists[p];
        if (list_id >= index.nlist) {
            continue;
        }

        const uint32_t begin = index.list_offsets[list_id];
        const uint32_t end = index.list_offsets[static_cast<size_t>(list_id) + 1];
        candidate_count += static_cast<uint64_t>(end - begin);

        for (uint32_t off = begin; off < end; ++off) {
            const uint32_t id = index.ids[off];
            const float *base_vec = index.vectors.data() + static_cast<size_t>(off) * index.vecdim;
            const float ip = inner_product_auto(base_vec, query, index.vecdim);
            push_topk(result, 1.0f - ip, id, k);
        }
    }

    return result;
}

static ResultHeap search_local_ivf_omp(const LocalIVFIndex &index,
                                       const float *query,
                                       const std::vector<uint32_t> &selected_lists,
                                       size_t k,
                                       int threads,
                                       uint64_t &candidate_count) {
    if (threads <= 1) {
        return search_local_ivf(index, query, selected_lists, k, candidate_count);
    }

    std::vector<ResultHeap> local_heaps(static_cast<size_t>(threads));
    std::vector<unsigned long long> local_candidates(static_cast<size_t>(threads), 0);

#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        ResultHeap local;
        unsigned long long candidates = 0;

#pragma omp for schedule(dynamic, 1)
        for (int p = 0; p < static_cast<int>(selected_lists.size()); ++p) {
            const uint32_t list_id = selected_lists[static_cast<size_t>(p)];
            if (list_id >= index.nlist) {
                continue;
            }

            const uint32_t begin = index.list_offsets[list_id];
            const uint32_t end = index.list_offsets[static_cast<size_t>(list_id) + 1];
            candidates += static_cast<unsigned long long>(end - begin);

            for (uint32_t off = begin; off < end; ++off) {
                const uint32_t id = index.ids[off];
                const float *base_vec = index.vectors.data() + static_cast<size_t>(off) * index.vecdim;
                const float ip = inner_product_auto(base_vec, query, index.vecdim);
                push_topk(local, 1.0f - ip, id, k);
            }
        }

        local_heaps[static_cast<size_t>(tid)] = std::move(local);
        local_candidates[static_cast<size_t>(tid)] = candidates;
    }

    candidate_count = 0;
    for (size_t i = 0; i < local_candidates.size(); ++i) {
        candidate_count += static_cast<uint64_t>(local_candidates[i]);
    }
    return merge_heaps(local_heaps, k);
}

static float evaluate_recall(ResultHeap result, const int *gt, size_t k) {
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert(static_cast<uint32_t>(gt[j]));
    }

    size_t acc = 0;
    while (!result.empty()) {
        const uint32_t id = result.top().second;
        result.pop();
        if (gtset.find(id) != gtset.end()) {
            ++acc;
        }
    }
    return static_cast<float>(acc) / static_cast<float>(k);
}

static ResultHeap merge_gathered_topk(const std::vector<float> &distances,
                                      const std::vector<uint32_t> &ids,
                                      size_t k) {
    ResultHeap merged;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        push_topk(merged, distances[i], ids[i], k);
    }
    return merged;
}

static void fill_topk_buffers(ResultHeap heap,
                              std::vector<float> &distances,
                              std::vector<uint32_t> &ids,
                              size_t k) {
    distances.assign(k, std::numeric_limits<float>::infinity());
    ids.assign(k, std::numeric_limits<uint32_t>::max());

    const std::vector<std::pair<float, uint32_t> > items = heap_to_sorted_items(heap, k);
    for (size_t i = 0; i < items.size() && i < k; ++i) {
        distances[i] = items[i].first;
        ids[i] = items[i].second;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Config cfg = parse_args(argc, argv, rank);
    omp_set_dynamic(0);
    omp_set_num_threads(cfg.threads);

    size_t base_number = 0;
    size_t vecdim = 0;
    float *base = LoadDataOrAbort<float>(
        cfg.data_path + "DEEP100K.base.100k.fbin", base_number, vecdim, rank);
    cfg.nlist = std::max<size_t>(1, std::min(cfg.nlist, base_number));
    cfg.nprobe = std::max<size_t>(1, std::min(cfg.nprobe, cfg.nlist));

    size_t test_number = 0;
    size_t query_dim = 0;
    size_t gt_rows = 0;
    size_t gt_d = 0;
    float *queries = NULL;
    int *gt = NULL;

    if (rank == 0) {
        queries = LoadDataOrAbort<float>(
            cfg.data_path + "DEEP100K.query.fbin", test_number, query_dim, rank);
        gt = LoadDataOrAbort<int>(
            cfg.data_path + "DEEP100K.gt.query.100k.top100.bin", gt_rows, gt_d, rank);
        if (query_dim != vecdim) {
            abort_all("query dimension does not match base dimension", rank);
        }
        if (gt_rows != test_number) {
            abort_all("ground-truth row count does not match query count", rank);
        }
        if (gt_d < cfg.k) {
            abort_all("ground-truth top-k is smaller than requested --k", rank);
        }
        test_number = std::min(test_number, cfg.max_queries);
    }

    unsigned long long meta[3];
    if (rank == 0) {
        meta[0] = static_cast<unsigned long long>(test_number);
        meta[1] = static_cast<unsigned long long>(gt_d);
        meta[2] = static_cast<unsigned long long>(vecdim);
    }
    MPI_Bcast(meta, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    test_number = static_cast<size_t>(meta[0]);
    gt_d = static_cast<size_t>(meta[1]);
    vecdim = static_cast<size_t>(meta[2]);

    std::vector<float> centroids;
    bool centroid_cache_hit = false;
    double centroid_build_ms = 0.0;
    const std::string cache_path = centroid_cache_path(cfg.nlist, vecdim, base_number);

    if (rank == 0) {
        const double t0 = MPI_Wtime();
        if (!cfg.rebuild_centroids && file_exists(cache_path)) {
            centroid_cache_hit = load_centroids(centroids, cache_path, cfg.nlist, vecdim, base_number);
        }
        if (!centroid_cache_hit) {
            std::cerr << "train IVF centroids: nlist=" << cfg.nlist
                      << " train_n=" << std::min(cfg.train_n, base_number)
                      << " iters=" << cfg.iters << "\n";
            train_ivf_centroids(base, base_number, vecdim, cfg.nlist, cfg.train_n, cfg.iters, centroids);
            if (!save_centroids(centroids, cache_path, cfg.nlist, vecdim, base_number)) {
                std::cerr << "warning: failed to save centroid cache " << cache_path << "\n";
            }
        } else {
            std::cerr << "loaded centroid cache " << cache_path << "\n";
        }
        centroid_build_ms = (MPI_Wtime() - t0) * 1000.0;
    } else {
        centroids.assign(cfg.nlist * vecdim, 0.0f);
    }

    int cache_hit_int = centroid_cache_hit ? 1 : 0;
    MPI_Bcast(&cache_hit_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    centroid_cache_hit = cache_hit_int != 0;
    MPI_Bcast(centroids.data(),
              mpi_count(centroids.size(), "centroids", rank),
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);

    size_t local_begin = 0;
    size_t local_end = 0;
    if (cfg.partition == "block-id") {
        partition_block(base_number, rank, world_size, local_begin, local_end);
    } else {
        local_begin = 0;
        local_end = base_number;
    }
    const std::string local_cache_path = local_ivf_cache_path(
        cfg.nlist, vecdim, base_number, world_size, rank, cfg.partition, local_begin, local_end);

    const double local_build_start = MPI_Wtime();
    LocalIVFIndex local_index;
    const bool local_cache_hit = !cfg.rebuild_centroids &&
                                 load_local_ivf_index(local_index,
                                                      local_cache_path,
                                                      cfg.nlist,
                                                      vecdim,
                                                      base_number,
                                                      world_size,
                                                      rank,
                                                      local_begin,
                                                      local_end);
    if (!local_cache_hit) {
        local_index = build_local_ivf_index(
            base, base_number, vecdim, centroids, cfg.nlist, rank, world_size, cfg.partition);
        if (!save_local_ivf_index(local_index, local_cache_path, base_number, world_size, rank)) {
            std::cerr << "rank " << rank
                      << " warning: failed to save local IVF cache "
                      << local_cache_path << "\n";
        }
    }
    const double local_build_ms = (MPI_Wtime() - local_build_start) * 1000.0;

    const unsigned long long local_work =
        static_cast<unsigned long long>(local_index.ids.size());
    std::vector<unsigned long long> all_work;
    std::vector<double> all_build_ms;
    std::vector<int> all_local_cache_hit;
    if (rank == 0) {
        all_work.assign(world_size, 0);
        all_build_ms.assign(world_size, 0.0);
        all_local_cache_hit.assign(world_size, 0);
    }
    const int local_cache_hit_int = local_cache_hit ? 1 : 0;
    MPI_Gather(&local_work, 1, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? all_work.data() : NULL, 1, MPI_UNSIGNED_LONG_LONG,
               0, MPI_COMM_WORLD);
    MPI_Gather(&local_build_ms, 1, MPI_DOUBLE,
               rank == 0 ? all_build_ms.data() : NULL, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    MPI_Gather(&local_cache_hit_int, 1, MPI_INT,
               rank == 0 ? all_local_cache_hit.data() : NULL, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cerr << "start IVF MPI search: np=" << world_size
                  << " threads_per_rank=" << cfg.threads
                  << " nlist=" << cfg.nlist
                  << " nprobe=" << cfg.nprobe
                  << " queries=" << test_number
                  << " k=" << cfg.k << "\n";
    }

    std::vector<float> query_buffer(vecdim, 0.0f);
    std::vector<uint32_t> selected_lists(cfg.nprobe, 0);
    std::vector<float> local_distances;
    std::vector<uint32_t> local_ids;
    std::vector<float> gathered_distances;
    std::vector<uint32_t> gathered_ids;
    std::vector<double> gathered_search_us;
    std::vector<unsigned long long> gathered_candidates;

    if (rank == 0) {
        gathered_distances.assign(static_cast<size_t>(world_size) * cfg.k, 0.0f);
        gathered_ids.assign(static_cast<size_t>(world_size) * cfg.k, 0);
        gathered_search_us.assign(world_size, 0.0);
        gathered_candidates.assign(world_size, 0);
    }

    QueryMetrics metrics;
    const double batch_start = MPI_Wtime();

    for (size_t qi = 0; qi < test_number; ++qi) {
        double query_start = 0.0;
        double bcast_us = 0.0;
        if (rank == 0) {
            query_start = MPI_Wtime();
            std::memcpy(query_buffer.data(), queries + qi * vecdim, sizeof(float) * vecdim);
            selected_lists = select_ivf_lists_omp(
                centroids, cfg.nlist, vecdim, query_buffer.data(), cfg.nprobe, cfg.threads);
        }

        const double bcast_start = MPI_Wtime();
        MPI_Bcast(query_buffer.data(),
                  mpi_count(vecdim, "query vector", rank),
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(selected_lists.data(),
                  mpi_count(cfg.nprobe, "selected lists", rank),
                  MPI_UNSIGNED,
                  0,
                  MPI_COMM_WORLD);
        if (rank == 0) {
            bcast_us = (MPI_Wtime() - bcast_start) * 1000000.0;
        }

        uint64_t local_candidates_u64 = 0;
        const double search_start = MPI_Wtime();
        ResultHeap local_result = search_local_ivf_omp(
            local_index, query_buffer.data(), selected_lists, cfg.k, cfg.threads, local_candidates_u64);
        const double local_search_us = (MPI_Wtime() - search_start) * 1000000.0;
        const unsigned long long local_candidates =
            static_cast<unsigned long long>(local_candidates_u64);
        fill_topk_buffers(local_result, local_distances, local_ids, cfg.k);

        const double gather_start = MPI_Wtime();
        MPI_Gather(local_distances.data(), mpi_count(cfg.k, "local distances", rank), MPI_FLOAT,
                   rank == 0 ? gathered_distances.data() : NULL,
                   mpi_count(cfg.k, "gathered distances", rank),
                   MPI_FLOAT,
                   0,
                   MPI_COMM_WORLD);
        MPI_Gather(local_ids.data(), mpi_count(cfg.k, "local ids", rank), MPI_UNSIGNED,
                   rank == 0 ? gathered_ids.data() : NULL,
                   mpi_count(cfg.k, "gathered ids", rank),
                   MPI_UNSIGNED,
                   0,
                   MPI_COMM_WORLD);
        const double gather_us = rank == 0 ? (MPI_Wtime() - gather_start) * 1000000.0 : 0.0;

        MPI_Gather(&local_search_us, 1, MPI_DOUBLE,
                   rank == 0 ? gathered_search_us.data() : NULL, 1, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        MPI_Gather(&local_candidates, 1, MPI_UNSIGNED_LONG_LONG,
                   rank == 0 ? gathered_candidates.data() : NULL, 1, MPI_UNSIGNED_LONG_LONG,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            const double merge_start = MPI_Wtime();
            ResultHeap merged = merge_gathered_topk(gathered_distances, gathered_ids, cfg.k);
            const double merge_us = (MPI_Wtime() - merge_start) * 1000000.0;
            const double latency_us = (MPI_Wtime() - query_start) * 1000000.0;
            const float recall = evaluate_recall(merged, gt + qi * gt_d, cfg.k);

            double search_sum = 0.0;
            double search_max = 0.0;
            unsigned long long candidate_sum = 0;
            for (int r = 0; r < world_size; ++r) {
                search_sum += gathered_search_us[r];
                search_max = std::max(search_max, gathered_search_us[r]);
                candidate_sum += gathered_candidates[r];
            }

            metrics.recall_sum += recall;
            metrics.latency_us_sum += latency_us;
            metrics.bcast_us_sum += bcast_us;
            metrics.local_search_avg_us_sum += search_sum / static_cast<double>(world_size);
            metrics.local_search_max_us_sum += search_max;
            metrics.gather_us_sum += gather_us;
            metrics.merge_us_sum += merge_us;
            metrics.candidate_count_sum += static_cast<double>(candidate_sum);
        }
    }

    const double batch_time_ms = (MPI_Wtime() - batch_start) * 1000.0;

    if (rank == 0) {
        unsigned long long work_min = all_work.empty() ? 0 : all_work[0];
        unsigned long long work_max = all_work.empty() ? 0 : all_work[0];
        unsigned long long work_sum = 0;
        double local_build_max_ms = 0.0;
        double local_build_avg_ms = 0.0;
        int local_cache_hits = 0;
        for (int r = 0; r < world_size; ++r) {
            work_min = std::min(work_min, all_work[r]);
            work_max = std::max(work_max, all_work[r]);
            work_sum += all_work[r];
            local_build_avg_ms += all_build_ms[r];
            local_build_max_ms = std::max(local_build_max_ms, all_build_ms[r]);
            local_cache_hits += all_local_cache_hit[r];
        }
        const double work_avg = static_cast<double>(work_sum) / static_cast<double>(world_size);
        local_build_avg_ms /= static_cast<double>(world_size);
        const double imbalance_ratio = work_avg > 0.0 ? static_cast<double>(work_max) / work_avg : 0.0;
        const double q = static_cast<double>(test_number);
        const double avg_recall = metrics.recall_sum / q;
        const double avg_latency = metrics.latency_us_sum / q;
        const double avg_bcast = metrics.bcast_us_sum / q;
        const double avg_search_avg = metrics.local_search_avg_us_sum / q;
        const double avg_search_max = metrics.local_search_max_us_sum / q;
        const double avg_gather = metrics.gather_us_sum / q;
        const double avg_merge = metrics.merge_us_sum / q;
        const double avg_candidates = metrics.candidate_count_sum / q;
        const double qps = batch_time_ms > 0.0 ? q * 1000.0 / batch_time_ms : 0.0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "algorithm: " << cfg.algo << "\n";
        std::cout << "run_id: " << cfg.run_id << "\n";
        std::cout << "group: " << cfg.group << "\n";
        std::cout << "partition: " << cfg.partition << "\n";
        std::cout << "np: " << world_size << "\n";
        std::cout << "threads_per_rank: " << cfg.threads << "\n";
        std::cout << "total_cores: " << world_size * cfg.threads << "\n";
        std::cout << "nodes: " << cfg.nodes << "\n";
        std::cout << "ppn: " << cfg.ppn << "\n";
        std::cout << "nlist: " << cfg.nlist << "\n";
        std::cout << "nprobe: " << cfg.nprobe << "\n";
        std::cout << "k: " << cfg.k << "\n";
        std::cout << "test_queries: " << test_number << "\n";
        std::cout << "train_n: " << cfg.train_n << "\n";
        std::cout << "iters: " << cfg.iters << "\n";
        std::cout << "centroid_cache_hit: " << (centroid_cache_hit ? 1 : 0) << "\n";
        std::cout << "centroid_build_time_ms: " << centroid_build_ms << "\n";
        std::cout << "local_index_cache_hits: " << local_cache_hits << "\n";
        std::cout << "local_index_build_avg_ms: " << local_build_avg_ms << "\n";
        std::cout << "local_index_build_max_ms: " << local_build_max_ms << "\n";
        std::cout << "work_items_min: " << work_min << "\n";
        std::cout << "work_items_avg: " << work_avg << "\n";
        std::cout << "work_items_max: " << work_max << "\n";
        std::cout << "imbalance_ratio: " << imbalance_ratio << "\n";
        std::cout << "average recall: " << avg_recall << "\n";
        std::cout << "average latency (us): " << avg_latency << "\n";
        std::cout << "query_bcast_us: " << avg_bcast << "\n";
        std::cout << "local_search_us_avg: " << avg_search_avg << "\n";
        std::cout << "local_search_us_max: " << avg_search_max << "\n";
        std::cout << "gather_us: " << avg_gather << "\n";
        std::cout << "merge_us: " << avg_merge << "\n";
        std::cout << "candidate_count_avg: " << avg_candidates << "\n";
        std::cout << "batch_time_ms: " << batch_time_ms << "\n";
        std::cout << "qps: " << qps << "\n";

        std::cout << "csv_header: run_id,group,algorithm,np,threads,nodes,ppn,total_cores,nlist,nprobe,k,test_queries,partition,cache_hit,recall,latency_us,bcast_us,search_avg_us,search_max_us,gather_us,merge_us,candidate_count_avg,work_min,work_avg,work_max,imbalance_ratio,batch_time_ms,qps\n";
        std::cout << "csv: "
                  << cfg.run_id << ","
                  << cfg.group << ","
                  << cfg.algo << ","
                  << world_size << ","
                  << cfg.threads << ","
                  << cfg.nodes << ","
                  << cfg.ppn << ","
                  << world_size * cfg.threads << ","
                  << cfg.nlist << ","
                  << cfg.nprobe << ","
                  << cfg.k << ","
                  << test_number << ","
                  << cfg.partition << ","
                  << (centroid_cache_hit ? 1 : 0) << ","
                  << avg_recall << ","
                  << avg_latency << ","
                  << avg_bcast << ","
                  << avg_search_avg << ","
                  << avg_search_max << ","
                  << avg_gather << ","
                  << avg_merge << ","
                  << avg_candidates << ","
                  << work_min << ","
                  << work_avg << ","
                  << work_max << ","
                  << imbalance_ratio << ","
                  << batch_time_ms << ","
                  << qps << "\n";
    }

    delete[] queries;
    delete[] gt;
    delete[] base;
    MPI_Finalize();
    return 0;
}
