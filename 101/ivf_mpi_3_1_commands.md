# IVF MPI 3.1 实验指令清单

在服务器主节点执行前，先进入 ANN 目录：

```bash
cd ~/ann
```

下面所有命令都使用 `submit.sh`，格式为：

```bash
sh submit.sh <nodes> <ppn> <np> <nlist> <nprobe> <test_queries>
```

不需要每次 `rm -rf main`。`submit.sh` 每次都会执行：

```bash
mpic++ -O2 -std=c++11 main.cc -o main
```

这个命令会直接覆盖旧的 `main` 可执行文件。



## 1. 强扩展实验

固定：

```text
nlist=512
nprobe=64
test_queries=2000
```

```bash
sh submit.sh 1 1 1 512 64 2000
```

```bash
sh submit.sh 1 2 2 512 64 2000
```

```bash
sh submit.sh 1 4 4 512 64 2000
```

```bash
sh submit.sh 1 8 8 512 64 2000
```

```bash
sh submit.sh 2 8 16 512 64 2000
```

```bash
sh submit.sh 4 8 32 512 64 2000
```

## 2. Recall-Latency 参数扫描

固定：

```text
nodes=1
ppn=4
np=4
test_queries=2000
```

扫描：

```text
nlist={256,512}
nprobe={8,16,32,64}
```

```bash
sh submit.sh 1 4 4 256 8 2000
```

```bash
sh submit.sh 1 4 4 256 16 2000
```

```bash
sh submit.sh 1 4 4 256 32 2000
```

```bash
sh submit.sh 1 4 4 256 64 2000
```

```bash
sh submit.sh 1 4 4 512 8 2000
```

```bash
sh submit.sh 1 4 4 512 16 2000
```

```bash
sh submit.sh 1 4 4 512 32 2000
```

```bash
sh submit.sh 1 4 4 512 64 2000
```


