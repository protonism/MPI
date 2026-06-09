# IVF MPI 3.1 perf 指令说明

本实验环境中 PBS 计算节点可能没有 `perf` 命令，因此 profiling 可以在登录节点对同一份 `main` 可执行程序进行。正式性能数据仍以 PBS 批处理实验结果为准，perf 结果用于补充分析硬件事件和热点函数。

## 1. 前置步骤：生成 main

进入实验目录：

```bash
cd ~/ann
```

如果当前目录还没有 `main`，或者刚更新过 `main.cc`，需要先生成可执行文件。最省事的方法是先跑一次 `submit.sh`，让脚本完成编译：

```bash
sh submit.sh 1 1 1 512 64 2000 10
```

这一步的主要目的可以只是生成/更新 `main`。如果 `files/ivf_mpi_results.csv` 表头曾经混乱，`submit.sh` 结尾打印的 `latest experiment result` 不一定可靠；perf 分析只依赖后续直接运行的 `./main` 或 `mpiexec ./main`。

也可以不通过 `submit.sh`，直接手动编译。建议带调试符号和 frame pointer，便于 `perf report` 展示函数栈：

```bash
mpic++ -O2 -g -fno-omit-frame-pointer -std=c++11 main.cc -o main
```

如果只跑 `perf stat`，普通 `-O2` 编译也可以：

```bash
mpic++ -O2 -std=c++11 main.cc -o main
```

确认 `main` 已存在：

```bash
ls -lh main
```

## 2. perf stat：np=1 baseline

用于测量单进程代表点的硬件计数。

```bash
perf stat -r 3 -e cycles,instructions,l1d_cache_refill,l1d_cache,l2d_cache_refill,l2d_cache -- \
./main --algo=ivf-mpi --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=1 --nodes=1 --ppn=1
```

截图时只保留最后的 `Performance counter stats` 表格即可，前面的程序日志不需要放入报告。

## 3. perf stat：np=8 单节点 MPI

如果登录节点允许本地 `mpiexec`，运行：

```bash
perf stat -r 3 -e cycles,instructions,l1d_cache_refill,l1d_cache,l2d_cache_refill,l2d_cache -- \
/usr/local/bin/mpiexec -np 8 ./main --algo=ivf-mpi --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=1 --nodes=1 --ppn=8
```

该配置用于和 `np=1` 对比，说明数据分片并行后 elapsed、IPC 和 cache refill 情况的变化。

## 4. perf record：np=8 热点采样

用于生成热点函数报告。

```bash
perf record -F 99 -g --call-graph fp -e cycles -o perf.data -- \
/usr/local/bin/mpiexec -np 8 ./main --algo=ivf-mpi --nlist=512 --nprobe=64 --test=2000 --k=10 --threads=1 --nodes=1 --ppn=8
```

打开交互式报告：

```bash
perf report -i perf.data
```

如果只想输出文本版前 50 行：

```bash
perf report --stdio -i perf.data | head -n 50
```

## 5. 重点关注

perf stat 表中重点摘录：

```text
cycles
instructions
IPC
l1d_cache_refill / l1d_cache
l2d_cache_refill / l2d_cache
seconds time elapsed
```

perf report 中重点关注：

```text
main
MPIC_Wait
MPIC_Recv
MPIDI_CH3I_Progress
PMPI_Bcast / MPIR_Bcast
PMPI_Gather / MPIR_Gather
```

这些指标足够说明：IVF MPI 在 `np=8` 时搜索任务被分摊，但 MPI 通信和同步开销已经成为端到端延迟中的重要组成部分。
