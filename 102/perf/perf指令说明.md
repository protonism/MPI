# IVF MPI+OpenMP 3.2 perf 指令说明

先用 `submit.sh` 让脚本完成编译和一次正常 PBS 提交，然后回到登录节点当前目录直接执行 `perf stat`、`perf record`、`perf report`。本文不采用交互式 PBS，也不依赖 PBS 节点文件。

正式性能数据仍以 PBS 重复实验结果为准；这里的 perf 只用于补充分析硬件事件、OpenMP 热点和小规模 MPI 调用热点。

进入实验目录：

```bash
cd ~/ann
```

如果刚更新过 `main.cc`，先生成 `main`。为了让 `perf report` 调用栈更清楚，建议在 `submit.sh` 后再手动带调试符号和 frame pointer 编译一次：

```bash
mpic++ -O2 -g -fno-omit-frame-pointer -std=c++11 -fopenmp -pthread main.cc -o main
```

perf 事件：

```text
cycles,instructions,l1d_cache_refill,l1d_cache,l2d_cache_refill,l2d_cache
```

注意：不要在登录节点直接跑 16 rank、2000 条 query、重复 3 次的 perf 组合。它会把所有 rank 都压在登录节点上，并且重复运行多遍，容易长时间停在 `start IVF MPI search`。下面保留登录节点可承受、最值得分析的参数组。

## 1. 单 rank 基线：np=1, threads=1

目的：得到没有 OpenMP 加速时的硬件事件基线，用来和 8 线程版本比较 IPC、cache refill 和热点函数。

```bash
sh submit.sh 1 1 1 512 64 2000 1 10
```

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE perf stat -r 3 -e cycles,instructions,l1d_cache_refill,l1d_cache,l2d_cache_refill,l2d_cache -- ./main --algo=ivf-mpi-omp --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=1 --nodes=1 --ppn=1 --run-id=perf_np1_t1
```

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE perf record -F 99 -g --call-graph fp -e cycles -o perf_np1_t1.data -- ./main --algo=ivf-mpi-omp --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=1 --nodes=1 --ppn=1 --run-id=perf_np1_t1
```

```bash
perf report -i perf_np1_t1.data
```

## 2. 单 rank OpenMP 代表点：np=1, threads=8

目的：证明 3.2 代码中的 OpenMP list scan 在单进程内确实有效，并观察 `search_local_ivf_omp`、`select_ivf_lists_omp`、`GOMP_parallel` 等热点。

```bash
sh submit.sh 1 8 1 512 64 2000 8 10
```

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE perf stat -r 3 -e cycles,instructions,l1d_cache_refill,l1d_cache,l2d_cache_refill,l2d_cache -- ./main --algo=ivf-mpi-omp --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=8 --nodes=1 --ppn=8 --run-id=perf_np1_t8
```

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE perf record -F 99 -g --call-graph fp -e cycles -o perf_np1_t8.data -- ./main --algo=ivf-mpi-omp --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=8 --nodes=1 --ppn=8 --run-id=perf_np1_t8
```

```bash
perf report -i perf_np1_t8.data
```

## 3. 登录节点小规模 MPI+OpenMP 参考：np=4, threads=2

目的：在不申请 PBS 交互资源的情况下，保留一个能看到 MPI 调用热点的小规模混合并行参考点。这里把 `test` 降到 500，并把 `perf stat` 的重复次数设为 1，避免登录节点长时间卡住。

```bash
sh submit.sh 1 8 4 512 64 500 2 10
```

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE perf stat -r 1 -e cycles,instructions,l1d_cache_refill,l1d_cache,l2d_cache_refill,l2d_cache -- /usr/local/bin/mpiexec -np 4 ./main --algo=ivf-mpi-omp --nlist=512 --nprobe=64 --test=500 --k=10 --threads=2 --nodes=1 --ppn=8 --run-id=perf_np4_t2_small
```

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE perf record -F 99 -g --call-graph fp -e cycles -o perf_np4_t2_small.data -- /usr/local/bin/mpiexec -np 4 ./main --algo=ivf-mpi-omp --nlist=512 --nprobe=64 --test=500 --k=10 --threads=2 --nodes=1 --ppn=8 --run-id=perf_np4_t2_small
```

```bash
perf report -i perf_np4_t2_small.data
```

## 4. 重点关注

```text
cycles
instructions
IPC
l1d_cache_refill / l1d_cache
l2d_cache_refill / l2d_cache
seconds time elapsed
```

`perf report` 重点查看：

```text
search_local_ivf_omp
select_ivf_lists_omp
inner_product_auto
inner_product_neon_96
GOMP_parallel
MPI_Bcast / PMPI_Bcast / MPIR_Bcast
MPI_Gather / PMPI_Gather / MPIR_Gather
MPIC_Wait / MPIDI_CH3I_Progress
```

三组命令的使用方式：

```text
np=1,threads=1：作为无 OpenMP 的硬件事件基线。
np=1,threads=8：说明 OpenMP 在单 rank 内部加速 list scan 的效果和线程开销。
np=4,threads=2,test=500：只作为登录节点可承受的小规模 MPI+OpenMP 热点参考，不替代正式 PBS 结果。
```
