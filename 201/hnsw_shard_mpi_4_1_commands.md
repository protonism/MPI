# HNSW Shard MPI 4.1 实验指令清单

本组选择 **数据分片 HNSW + MPI top-k merge** 作为图索引 MPI 主线：每个 rank 负责一片 base data，独立建立本地 HNSW，查询时广播 query，各 rank 返回本地 top-k，最后由 rank 0 合并全局 top-k。通信量稳定，阶段耗时容易拆分，适合和 101/102/103 的 IVF MPI 结果对照分析。

在服务器主节点进入 ANN 目录：

```bash
cd ~/ann
```

提交格式：

```bash
sh submit.sh <nodes> <ppn> <np> <efSearch> <M> <efConstruction> <test_queries> <partition> <group>
```

可用 `partition`：

```text
block-id
cyclic
random-hash
```

结果文件：

```text
files/hnsw_shard_mpi_results.csv
```

CSV 使用追加写入，并记录 `run_id`、`group`、`partition`、`M`、`efConstruction`、`efSearch`、构建时间和通信/搜索/合并耗时。



## 1. HNSW-A：单进程 baseline 曲线

固定：

```text
np=1
M=16
efConstruction=150
test_queries=2000
partition=block-id
```

扫描：

```text
efSearch={20,50,80,100,150,200}
```

参数组：

```bash
sh submit.sh 1 1 1 20 16 150 2000 block-id HNSW-A
sh submit.sh 1 1 1 50 16 150 2000 block-id HNSW-A
sh submit.sh 1 1 1 80 16 150 2000 block-id HNSW-A
sh submit.sh 1 1 1 100 16 150 2000 block-id HNSW-A
sh submit.sh 1 1 1 150 16 150 2000 block-id HNSW-A
sh submit.sh 1 1 1 200 16 150 2000 block-id HNSW-A
```

## 2. HNSW-B：数据分片 HNSW 强扩展

固定：

```text
M=16
efConstruction=150
efSearch=100
test_queries=2000
partition=block-id
```

扫描：

```text
np={1,2,4,8,16}
```

参数组：

```bash
sh submit.sh 1 1 1 100 16 150 2000 block-id HNSW-B
sh submit.sh 1 2 2 100 16 150 2000 block-id HNSW-B
sh submit.sh 1 4 4 100 16 150 2000 block-id HNSW-B
sh submit.sh 1 8 8 100 16 150 2000 block-id HNSW-B
sh submit.sh 2 8 16 100 16 150 2000 block-id HNSW-B
```

## 3. HNSW-C：Recall-Latency 折中

固定：

```text
nodes=1
ppn=4
np=4
M=16
efConstruction=150
test_queries=2000
partition=block-id
```

扫描：

```text
efSearch={50,80,100,150,200}
```

参数组：

```bash
sh submit.sh 1 4 4 50 16 150 2000 block-id HNSW-C
sh submit.sh 1 4 4 80 16 150 2000 block-id HNSW-C
sh submit.sh 1 4 4 100 16 150 2000 block-id HNSW-C
sh submit.sh 1 4 4 150 16 150 2000 block-id HNSW-C
sh submit.sh 1 4 4 200 16 150 2000 block-id HNSW-C
```

## 4. HNSW-D：分片方式影响

固定：

```text
nodes=1
ppn=4
np=4
M=16
efConstruction=150
efSearch=100
test_queries=2000
```

扫描：

```text
partition={block-id,cyclic,random-hash}
```

参数组：

```bash
sh submit.sh 1 4 4 100 16 150 2000 block-id HNSW-D
sh submit.sh 1 4 4 100 16 150 2000 cyclic HNSW-D
sh submit.sh 1 4 4 100 16 150 2000 random-hash HNSW-D
```

## 5. HNSW-E：构建参数影响

固定：

```text
nodes=1
ppn=4
np=4
efSearch=100
test_queries=2000
partition=block-id
```

扫描：

```text
(M,efConstruction)={(8,100),(8,150),(16,100),(16,150),(16,200),(24,150),(32,150)}
```

参数组：

```bash
sh submit.sh 1 4 4 100 8 100 2000 block-id HNSW-E
sh submit.sh 1 4 4 100 8 150 2000 block-id HNSW-E
sh submit.sh 1 4 4 100 16 100 2000 block-id HNSW-E
sh submit.sh 1 4 4 100 16 150 2000 block-id HNSW-E
sh submit.sh 1 4 4 100 16 200 2000 block-id HNSW-E
sh submit.sh 1 4 4 100 24 150 2000 block-id HNSW-E
sh submit.sh 1 4 4 100 32 150 2000 block-id HNSW-E
```

## 6. HNSW-F：同 np 跨节点通信影响

固定：

```text
np=8
M=16
efConstruction=150
efSearch=100
test_queries=2000
partition=block-id
```

扫描：

```text
(nodes,ppn)=(1,8),(2,4),(4,2)
```

参数组：

```bash
sh submit.sh 1 8 8 100 16 150 2000 block-id HNSW-F
sh submit.sh 2 4 8 100 16 150 2000 block-id HNSW-F
sh submit.sh 4 2 8 100 16 150 2000 block-id HNSW-F
```

## 7. 查看和补齐结果

查看 CSV：

```bash
tail -n 10 files/hnsw_shard_mpi_results.csv
cat files/hnsw_shard_mpi_results.csv
```

如果某些 job 的日志已经产生，但 CSV 缺行，可从日志中只追加缺失行；已有 CSV 行不会被覆盖：

```bash
sh prepend_out_logs_to_hnsw_csv.sh
```
