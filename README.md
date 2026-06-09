# ANN 的 MPI 优化

本项目是在已有 SIMD、OpenMP 和 Pthread 优化基础上，对近似最近邻搜索（Approximate Nearest Neighbor Search，ANNS）查询阶段进行 MPI 并行化的实验代码。实验主要围绕 IVF-SIMD 和 HNSW 两类索引展开，重点分析进程级数据分片、局部 top-k 搜索、MPI 通信、全局结果归并、MPI+OpenMP 混合并行以及负载均衡策略对召回率和查询延迟的影响。



## 1. 运行环境

实验代码面向 openEuler 服务器环境，使用 PBS 作业调度系统提交任务。运行前需要确保服务器已安装：

- MPI 编译器与运行时：`mpic++`、`mpiexec`
- PBS 命令：`qsub`、`qstat`、`qdel`
- OpenMP 支持：用于 `102` 和 `103`
- ARM NEON 指令支持：用于 IVF-SIMD 内积计算优化
- `hnswlib`：用于 `201` 图索引实验

## 2. 服务器运行版本

最终提交至 openEuler 实验平台的版本为 `103` 文件夹中的 IVF 综合实验版本。进入该目录后执行：

```bash
sh submit.sh 1 8 8 512 64 2000 1 block-id IVF-D
```

表示：申请 1 个节点，每个节点 8 个核心，启动 8 个 MPI 进程；IVF 使用 512 个倒排列表，每条 query 扫描其中 64 个列表；测试 2000 条 query；每个 MPI rank 使用 1 个线程；采用按全局 id 连续分块的 `block-id` 策略，并将结果记录为 `IVF-C` 实验组。

通用格式为：

```bash
sh submit.sh <nodes> <ppn> <np> <nlist> <nprobe> <test_queries> <threads_per_rank> <partition> <group>
```

参数含义如下：

| 参数 | 含义 |
| --- | --- |
| `nodes` | 申请的计算节点数量 |
| `ppn` | 每个节点申请的处理器核心数（processors per node） |
| `np` | 启动的 MPI 进程总数 |
| `nlist` | IVF 倒排列表数量，即聚类中心数量 |
| `nprobe` | 每条查询需要扫描的倒排列表数量 |
| `test_queries` | 测试查询向量数量 |
| `threads_per_rank` | 每个 MPI rank 内部使用的 OpenMP 线程数 |
| `partition` | 数据分片策略，可选 `block-id`、`contiguous-list`、`greedy-list` |
| `group` | 实验分组标签，会写入 CSV，便于后续筛选和绘图 |

资源配置需要满足：

```text
np * threads_per_rank <= nodes * ppn
nodes <= 4
ppn <= 8
```

## 3. 目录结构

```text
.
├── 101/                     # IVF-SIMD 纯 MPI 基础实现
├── 102/                     # IVF-SIMD 的 MPI+OpenMP 混合并行实现
├── 103/                     # IVF 综合实验版本：混合并行、分片策略与完整实验组
├── 201/                     # 数据分片 HNSW 的 MPI 图索引尝试
└── README.md
```

### 3.1 `101`：IVF-SIMD 纯 MPI 基础实现

`101` 是 IVF 算法的第一版 MPI 实现。base 向量按照全局 id 的连续区间划分到不同 rank，每个 rank 仅维护本地 IVF 索引。查询阶段由 rank 0 完成 coarse search，选择距离 query 最近的 `nprobe` 个倒排列表，然后通过 `MPI_Bcast` 广播 query 和列表编号。各 rank 在本地分片中调用 SIMD 优化的内积计算函数扫描候选，维护本地 top-k，最后通过 `MPI_Gather` 收集候选，并由 rank 0 完成全局 top-k merge。

该版本主要用于分析：

- 纯 MPI 的单节点强扩展效果；
- 跨节点后 `Bcast`、`Gather` 和同步等待带来的通信瓶颈；
- `nlist`、`nprobe` 对 Recall-Latency 折中的影响；
- `perf stat` 与 `perf record` 下的缓存、IPC 和 MPI 通信热点。

### 3.2 `102`：IVF-SIMD 的 MPI+OpenMP 混合并行实现

`102` 在 `101` 基础上加入 OpenMP 进程内并行。MPI 负责 rank 之间的数据分片，OpenMP 负责每个 rank 内部的 coarse search 和 selected lists 扫描。

其中，coarse search 使用静态调度；倒排列表扫描使用 `schedule(dynamic, 1)`，以减轻不同 list 长度差异导致的线程负载不均。每个 OpenMP 线程维护独立的局部堆，线程并行区结束后先在 rank 内合并，再参与 MPI 层面的全局 top-k merge。

该版本主要用于比较：

- 单 rank 内线程数增加后的局部搜索收益；
- 固定总核心数时，不同 `np × threads_per_rank` 组合的性能差异；
- OpenMP runtime、线程调度、同步等待和 MPI 通信共同造成的开销。

### 3.3 `103`：IVF 综合调参与负载均衡策略

`103` 是本项目中 IVF 部分的最终版本，也是当前推荐提交到 openEuler 服务器运行的版本。该版本继承了 `102` 的 MPI+OpenMP 混合并行能力，并补充了三种数据分片策略：

```text
block-id
contiguous-list
greedy-list
```

三种策略的含义如下：

| 策略 | 含义 |
| --- | --- |
| `block-id` | 按全局 base id 连续分块，索引组织简单，局部访问连续 |
| `contiguous-list` | 按倒排列表编号连续分配到各 rank |
| `greedy-list` | 根据 list 大小进行贪心分配，优先将较长 list 分给当前负载较小的 rank |

该版本用于完成：

- 单进程正确性对齐；
- Recall-Latency 参数扫描；
- 纯 MPI 强扩展；
- 固定 rank 数下的跨节点通信影响分析；
- 固定总核心数下的纯 MPI 与混合并行比较；
- 不同分片策略下的延迟与负载均衡对比。

实验结果表明，更复杂的理论负载均衡策略不一定带来更低的端到端延迟。对于 DEEP100K 规模的数据，`block-id` 已经具有较好的均衡性，并且能够保持更好的连续内存访问和缓存局部性。

### 3.4 `201`：数据分片 HNSW 的 MPI 图索引尝试

`201` 是对图索引算法的 MPI 并行化尝试。该版本采用 data-shard HNSW 方案：每个 rank 根据分片策略持有一部分 base data，在本地独立建立 HNSW 索引；查询时 rank 0 广播 query，各 rank 调用本地 `searchKnn` 返回局部 top-k，最后由 rank 0 合并为全局 top-k。

可用分片策略为：

```text
block-id
cyclic
random-hash
```

通用运行格式为：

```bash
sh submit.sh <nodes> <ppn> <np> <efSearch> <M> <efConstruction> <test_queries> <partition> <group>
```

示例：

```bash
sh submit.sh 1 8 8 100 16 150 2000 block-id HNSW-B
```

该版本主要用于研究：

- `efSearch` 对召回率与延迟的影响；
- HNSW 分片数量增加后的强扩展表现；
- 跨节点运行时 `Gather+Merge` 通信开销的放大；
- `M`、`efConstruction` 和分片策略对索引构建与查询性能的影响。

## 4. 文件说明

每个实验文件夹均包含核心代码、提交脚本、实验命令说明和绘图相关文件。主要文件如下：

| 文件 | 作用 |
| --- | --- |
| `main.cc` | 当前实验版本的核心 C++ 实现 |
| `submit.sh` | 用户侧入口脚本：编译 `main.cc`、检查资源约束、调用 `qsub` 提交 PBS 作业、等待任务结束并输出最新结果摘要 |
| `qsub_mpi.sh` | PBS 作业脚本：读取 PBS 分配的节点列表，将可执行文件和缓存文件分发到各节点，调用 `mpiexec` 启动 MPI 程序，保存运行日志，并将 CSV 结果追加回主节点 |
| `*_commands.md` | 当前版本对应的实验指令清单，包含参数扫描、强扩展、跨节点、分片方式等实验组 |
| `rawdata.csv` | 从服务器收集并整理的原始实验数据，用于后续清洗、汇总与绘图 |
| `charts/` | 绘图脚本、整理后的 CSV、生成的图表和图表说明文档 |
| `charts/chart_captions*.md` | 对每张图的用途、横纵轴、对比关系和分析重点进行说明 |
| `perf/` | `101` 和 `102` 中的性能分析截图与说明文档 |

## 5. 提交脚本说明

以 `103` 为例，执行：

```bash
sh submit.sh 1 8 8 512 64 2000 1 block-id IVF-D
```

脚本会依次完成：

1. 使用 `mpic++` 编译 `main.cc`；
2. 检查 `np * threads_per_rank` 是否超过申请的总核心数；
3. 通过 `qsub` 提交 PBS 作业；
4. 在作业执行期间使用 `qstat` 轮询任务状态；
5. 完成后读取 CSV 最新一行，输出召回率、平均延迟、QPS、广播耗时、Gather 耗时、Merge 耗时、本地搜索耗时和负载不均衡比例等指标。

`qsub_mpi.sh` 在 PBS 计算节点上执行，主要负责：

1. 根据 `$PBS_NODEFILE` 获取节点列表；
2. 将主节点上的 `main` 可执行文件复制到各计算节点；
3. 同步已有索引缓存文件；
4. 使用 `/usr/local/bin/mpiexec` 启动 MPI 程序；
5. 将运行日志保存到 `files/`；
6. 将本次实验生成的 CSV 结果追加到主节点结果文件中；
7. 回收并同步新的索引缓存文件。

## 6. 结果文件

各版本默认将实验结果追加写入以下 CSV 文件：

```text
101: files/ivf_mpi_results.csv
102: files/ivf_mpi_omp_results.csv
103: files/ivf_mpi_3_3_results.csv
201: files/hnsw_shard_mpi_results.csv
```

常用查看命令：

```bash
tail -n 10 files/ivf_mpi_3_3_results.csv
cat files/ivf_mpi_3_3_results.csv
```

CSV 中记录了 Recall@k、平均查询延迟、QPS、广播耗时、本地搜索耗时、Gather 耗时、Merge 耗时、候选数量、不同 rank 的工作量以及负载不均衡比例等指标。

## 7. 绘图

各文件夹的 `charts/` 目录中包含绘图脚本和图表说明。以 `103` 为例：

```bash
cd 103/charts
python3 plot_ivf_mpi_3_3.py
```

生成的图表用于分析 Recall-Latency 折中、强扩展、跨节点通信开销、固定核心数下的混合并行效果，以及不同分片策略下的负载均衡表现。

## 8. 项目结论概述

本项目的实验结果表明：

1. IVF-SIMD 的 MPI 进程级分片能够明显降低每个 rank 的候选扫描量，在单节点内具有较好的加速效果；
2. 跨节点运行时，`MPI_Bcast`、`MPI_Gather` 和同步等待会显著放大，成为端到端延迟的重要瓶颈；
3. MPI+OpenMP 可以加速 rank 内部扫描，但固定总核心数时不一定优于纯 MPI；
4. 更复杂的负载均衡策略不一定更快，缓存局部性和同步成本同样重要；
5. HNSW 分片 MPI 可以保持较高召回率，但随着 rank 数增加，候选收集与全局 Merge 更容易成为瓶颈。

## 9. 备注

- `103` 是 IVF 部分的最终提交版本；
- `201` 是图索引方向的探索性实现；
- 每个版本更完整的批量测试命令请参考对应的 `*_commands.md` 文件。
