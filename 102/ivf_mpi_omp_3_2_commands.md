# IVF MPI+OpenMP 3.2 实验指令清单

在服务器主节点进入 ANN 目录：
```bash
cd ~/ann
```

先确认 `102` 目录里的 `main.cc`、`submit.sh`、`qsub_mpi.sh` 已复制到 `~/ann`。本版本使用 3.2 专属命名：

- 算法名：`ivf-mpi-omp`
- 结果文件：`files/ivf_mpi_omp_results.csv`
- 单次日志：`files/ivf_mpi_omp_<jobid>.out`

提交格式：
```bash
sh submit.sh <nodes> <ppn> <np> <nlist> <nprobe> <test_queries> <threads_per_rank>
```

资源约束：
```text
np * threads_per_rank <= nodes * ppn
nodes <= 4
ppn <= 8
```

正式统计时每条命令建议重复 3 次；这里每个参数组只列一次。CSV 会追加写入，并用 `run_id` 记录 PBS job id，重复测试不会覆盖旧记录。

## 0. 快速确认

只用于确认编译、OpenMP、qsub 和 CSV 追加正常，不计入正式统计：
```bash
sh submit.sh 1 2 1 512 64 200 2
```

## 1. 3.2 核心对照：总核心数固定为 16

固定：
```text
nlist=512
nprobe=64
test_queries=2000
total_cores=16
```

参数组：
```bash
sh submit.sh 2 8 16 512 64 2000 1
sh submit.sh 2 8 8 512 64 2000 2
sh submit.sh 2 8 4 512 64 2000 4
sh submit.sh 2 8 2 512 64 2000 8
```

## 2. 单进程 OpenMP 线程数影响

用于观察每个 rank 内 OpenMP list scan 的收益和线程开销。

固定：
```text
nodes=1
nlist=512
nprobe=64
test_queries=2000
np=1
```

参数组：
```bash
sh submit.sh 1 1 1 512 64 2000 1
sh submit.sh 1 2 1 512 64 2000 2
sh submit.sh 1 4 1 512 64 2000 4
sh submit.sh 1 8 1 512 64 2000 8
```

## 3. 固定 MPI 进程数，扫描 rank 内线程数

用于和 3.1 的纯 MPI 结果对比，观察 `np=4` 时每个 rank 内加线程是否值得。

固定：
```text
nlist=512
nprobe=64
test_queries=2000
np=4
```

参数组：
```bash
sh submit.sh 1 4 4 512 64 2000 1
sh submit.sh 1 8 4 512 64 2000 2
sh submit.sh 2 8 4 512 64 2000 4
```

## 4. Recall-Latency 参数扫描

固定：
```text
nodes=2
ppn=8
np=8
threads_per_rank=2
test_queries=2000
```

扫描：
```text
nlist={256,512}
nprobe={8,16,32,64}
```

参数组：
```bash
sh submit.sh 2 8 8 256 8 2000 2
sh submit.sh 2 8 8 256 16 2000 2
sh submit.sh 2 8 8 256 32 2000 2
sh submit.sh 2 8 8 256 64 2000 2
sh submit.sh 2 8 8 512 8 2000 2
sh submit.sh 2 8 8 512 16 2000 2
sh submit.sh 2 8 8 512 32 2000 2
sh submit.sh 2 8 8 512 64 2000 2
```

## 5. 查看结果

```bash
tail -n 10 files/ivf_mpi_omp_results.csv
cat files/ivf_mpi_omp_results.csv
```

如果需要从已有 job 日志重建或补齐 CSV：
```bash
sh prepend_out_logs_to_ivf_omp_csv.sh
```
