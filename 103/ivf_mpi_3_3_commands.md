# IVF MPI 3.3 实验指令清单

在服务器主节点进入 ANN 目录：

```bash
cd ~/ann
```

提交格式：

```bash
sh submit.sh <nodes> <ppn> <np> <nlist> <nprobe> <test_queries> <threads_per_rank> <partition> <group>
```

可用 `partition`：

```text
block-id
contiguous-list
greedy-list
```

资源约束：

```text
np * threads_per_rank <= nodes * ppn
nodes <= 4
ppn <= 8
```

结果文件：

```text
files/ivf_mpi_3_3_results.csv
```

CSV 使用追加写入，并记录 `run_id`、`group`、`partition`。

## 0. 快速确认

只用于确认编译、PBS、CSV 追加和分区参数可用，不计入正式统计。

```bash
sh submit.sh 1 2 1 512 64 200 2 block-id smoke
```

## 1. IVF-A：串行/MPI 正确性对齐

固定：

```text
np=1
threads_per_rank=1
k=10
test_queries=2000
partition=block-id
```

参数组：

```bash
sh submit.sh 1 1 1 512 32 2000 1 block-id IVF-A
sh submit.sh 1 1 1 512 64 2000 1 block-id IVF-A
sh submit.sh 1 1 1 256 32 2000 1 block-id IVF-A
sh submit.sh 1 1 1 256 64 2000 1 block-id IVF-A
```

## 2. IVF-B：Recall-Latency 曲线

固定：

```text
nodes=1
ppn=4
np=4
threads_per_rank=1
test_queries=2000
partition=block-id
```

扫描：

```text
nlist={256,512}
nprobe={8,16,32,64}
```

参数组：

```bash
sh submit.sh 1 4 4 256 8 2000 1 block-id IVF-B
sh submit.sh 1 4 4 256 16 2000 1 block-id IVF-B
sh submit.sh 1 4 4 256 32 2000 1 block-id IVF-B
sh submit.sh 1 4 4 256 64 2000 1 block-id IVF-B
sh submit.sh 1 4 4 512 8 2000 1 block-id IVF-B
sh submit.sh 1 4 4 512 16 2000 1 block-id IVF-B
sh submit.sh 1 4 4 512 32 2000 1 block-id IVF-B
sh submit.sh 1 4 4 512 64 2000 1 block-id IVF-B
```

## 3. IVF-C：纯 MPI 强扩展

固定：

```text
nlist=512
nprobe=64
threads_per_rank=1
test_queries=2000
partition=block-id
```

扫描：

```text
np={1,2,4,8,16,32}
```

参数组：

```bash
sh submit.sh 1 1 1 512 64 2000 1 block-id IVF-C
sh submit.sh 1 2 2 512 64 2000 1 block-id IVF-C
sh submit.sh 1 4 4 512 64 2000 1 block-id IVF-C
sh submit.sh 1 8 8 512 64 2000 1 block-id IVF-C
sh submit.sh 2 8 16 512 64 2000 1 block-id IVF-C
sh submit.sh 4 8 32 512 64 2000 1 block-id IVF-C
```

## 4. IVF-D：同 np 跨节点通信影响

固定：

```text
np=8
threads_per_rank=1
nlist=512
nprobe=64
test_queries=2000
partition=block-id
```

扫描：

```text
(nodes,ppn)=(1,8),(2,4),(4,2)
```

参数组：

```bash
sh submit.sh 1 8 8 512 64 2000 1 block-id IVF-D
sh submit.sh 2 4 8 512 64 2000 1 block-id IVF-D
sh submit.sh 4 2 8 512 64 2000 1 block-id IVF-D
```

## 5. IVF-E：混合并行总核数对照

固定：

```text
total_cores=16
nlist=512
nprobe=64
test_queries=2000
partition=block-id
```

扫描：

```text
(np,threads_per_rank)=(16,1),(8,2),(4,4),(2,8)
```

参数组：

```bash
sh submit.sh 2 8 16 512 64 2000 1 block-id IVF-E
sh submit.sh 2 8 8 512 64 2000 2 block-id IVF-E
sh submit.sh 2 8 4 512 64 2000 4 block-id IVF-E
sh submit.sh 2 8 2 512 64 2000 8 block-id IVF-E
```

## 6. IVF-F：负载均衡策略

固定：

```text
nodes=1
ppn=8
np=8
threads_per_rank=1
nlist=512
nprobe=64
test_queries=2000
```

扫描：

```text
partition={block-id,contiguous-list,greedy-list}
```

参数组：

```bash
sh submit.sh 1 8 8 512 64 2000 1 block-id IVF-F
sh submit.sh 1 8 8 512 64 2000 1 contiguous-list IVF-F
sh submit.sh 1 8 8 512 64 2000 1 greedy-list IVF-F
```

## 7. 查看和补齐结果

查看 CSV：

```bash
tail -n 10 files/ivf_mpi_3_3_results.csv
cat files/ivf_mpi_3_3_results.csv
```

如果某些 job 的日志已经产生，但 CSV 缺行，可从日志重建并去重：

```bash
sh prepend_out_logs_to_ivf_3_3_csv.sh
```
