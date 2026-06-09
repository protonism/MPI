#!/bin/sh
#PBS -N ann_hnsw_shard_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=8

: "${NP:=8}"
: "${EF_SEARCH:=100}"
: "${M:=16}"
: "${EFC:=150}"
: "${TEST_QUERIES:=2000}"
: "${K:=10}"
: "${THREADS:=1}"
: "${NODES_REQ:=1}"
: "${PPN_REQ:=8}"
: "${ALGO:=hnsw-shard-mpi}"
: "${PARTITION:=block-id}"
: "${GROUP:=manual}"
: "${RUN_ID:=${PBS_JOBID}}"

NODES=$(cat $PBS_NODEFILE | sort | uniq)
RESULT_DIR=/home/${USER}/files
RUN_LOG=${RESULT_DIR}/hnsw_shard_mpi_${PBS_JOBID}.out
MASTER_RESULT_DIR=/home/${USER}/ann/files
MASTER_RESULT_CSV=${MASTER_RESULT_DIR}/hnsw_shard_mpi_results.csv

for node in $NODES; do
    ssh ${node} "mkdir -p /home/${USER}/files" 1>&2
    scp master_ubss1:/home/${USER}/ann/main ${node}:/home/${USER}/main 1>&2
    if [ -d /home/${USER}/ann/files ]; then
        scp master_ubss1:/home/${USER}/ann/files/hnsw_shard_*.index ${node}:/home/${USER}/files/ 2>/dev/null || true
    fi
done

mkdir -p ${RESULT_DIR}
cd /home/${USER}
/usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/main --algo=${ALGO} --ef=${EF_SEARCH} --M=${M} --efc=${EFC} --test=${TEST_QUERIES} --k=${K} --threads=${THREADS} --nodes=${NODES_REQ} --ppn=${PPN_REQ} --partition=${PARTITION} --group=${GROUP} --run-id=${RUN_ID} | tee ${RUN_LOG}

HEADER=$(grep '^csv_header: ' ${RUN_LOG} | tail -n 1 | sed 's/^csv_header: //')
CSV_LINE=$(grep '^csv: ' ${RUN_LOG} | tail -n 1 | sed 's/^csv: //')

if [ -n "${CSV_LINE}" ]; then
    ssh master_ubss1 "mkdir -p ${MASTER_RESULT_DIR}; lock=${MASTER_RESULT_DIR}/.hnsw_shard_mpi_results.lock; while ! mkdir \"\$lock\" 2>/dev/null; do sleep 1; done; trap 'rmdir \"\$lock\"' EXIT; if [ ! -s ${MASTER_RESULT_CSV} ] && [ -n '${HEADER}' ]; then printf '%s\n' '${HEADER}' >> ${MASTER_RESULT_CSV}; fi; printf '%s\n' '${CSV_LINE}' >> ${MASTER_RESULT_CSV}" 2>&1
    echo "compact_csv: ${CSV_LINE}"
    echo "result_csv: ${MASTER_RESULT_CSV}"
else
    echo "warning: no csv line found in ${RUN_LOG}; check test.o and test.e"
fi

ssh master_ubss1 "mkdir -p ${MASTER_RESULT_DIR}" 1>&2
scp ${RUN_LOG} master_ubss1:${MASTER_RESULT_DIR}/ 2>&1

for node in $NODES; do
    scp ${node}:/home/${USER}/files/hnsw_shard_*.index master_ubss1:${MASTER_RESULT_DIR}/ 2>/dev/null || true
done
