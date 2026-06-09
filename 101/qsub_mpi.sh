#!/bin/sh
#PBS -N ann_ivf_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=8

: "${NP:=8}"
: "${NLIST:=512}"
: "${NPROBE:=64}"
: "${TEST_QUERIES:=2000}"
: "${K:=10}"
: "${THREADS:=1}"
: "${NODES_REQ:=1}"
: "${PPN_REQ:=8}"
: "${ALGO:=ivf-mpi}"
: "${PERF:=0}"
: "${PERF_SET:=arm}"
: "${PERF_REPEAT:=3}"
: "${PERF_FREQ:=99}"
: "${PERF_REPORT_LINES:=80}"

case "${PERF_SET}" in
    arm)
        PERF_EVENTS="task-clock,cycles,instructions,context-switches,cpu-migrations,l1d_cache,l1d_cache_refill,l2d_cache,l2d_cache_refill"
        ;;
    generic)
        PERF_EVENTS="task-clock,cycles,instructions,context-switches,cpu-migrations,cache-misses,L1-dcache-load-misses"
        ;;
    *)
        PERF_EVENTS="${PERF_SET}"
        ;;
esac

NODES=$(cat $PBS_NODEFILE | sort | uniq)
RESULT_DIR=/home/${USER}/files
RUN_LOG=${RESULT_DIR}/ivf_mpi_${PBS_JOBID}.out
MASTER_RESULT_DIR=/home/${USER}/ann/files
MASTER_RESULT_CSV=${MASTER_RESULT_DIR}/ivf_mpi_results.csv
PERF_DATA=${RESULT_DIR}/ivf_mpi_${PBS_JOBID}_perf.data
PERF_REPORT=${RESULT_DIR}/ivf_mpi_${PBS_JOBID}_perf_report.txt
LATEST_PERF_DATA=${RESULT_DIR}/ivf_mpi_latest_perf.data
LATEST_PERF_REPORT=${RESULT_DIR}/ivf_mpi_latest_perf_report.txt
LATEST_RUN_LOG=${RESULT_DIR}/ivf_mpi_latest_perf.out

for node in $NODES; do
    ssh ${node} "mkdir -p /home/${USER}/files" 1>&2
    scp master_ubss1:/home/${USER}/ann/main ${node}:/home/${USER}/main 1>&2
    if [ -d /home/${USER}/ann/files ]; then
        scp master_ubss1:/home/${USER}/ann/files/ivf_centroids_*.bin ${node}:/home/${USER}/files/ 2>/dev/null || true
        scp master_ubss1:/home/${USER}/ann/files/ivf_mpi_local_*.bin ${node}:/home/${USER}/files/ 2>/dev/null || true
    fi
done

mkdir -p ${RESULT_DIR}
cd /home/${USER}

case "${PERF}" in
    0|off|false|no)
        /usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/main --algo=${ALGO} --nlist=${NLIST} --nprobe=${NPROBE} --test=${TEST_QUERIES} --k=${K} --threads=${THREADS} --nodes=${NODES_REQ} --ppn=${PPN_REQ} | tee ${RUN_LOG}
        ;;
    1|stat)
        {
            echo "perf_mode: stat"
            echo "perf_events: ${PERF_EVENTS}"
            echo "perf_repeat: ${PERF_REPEAT}"
        } | tee ${RUN_LOG}
        perf stat -r ${PERF_REPEAT} -e ${PERF_EVENTS} -- /usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/main --algo=${ALGO} --nlist=${NLIST} --nprobe=${NPROBE} --test=${TEST_QUERIES} --k=${K} --threads=${THREADS} --nodes=${NODES_REQ} --ppn=${PPN_REQ} 2>&1 | tee -a ${RUN_LOG}
        ;;
    record)
        {
            echo "perf_mode: record"
            echo "perf_freq: ${PERF_FREQ}"
            echo "perf_data: ${PERF_DATA}"
        } | tee ${RUN_LOG}
        perf record -F ${PERF_FREQ} -g --call-graph fp -o ${PERF_DATA} -- /usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/main --algo=${ALGO} --nlist=${NLIST} --nprobe=${NPROBE} --test=${TEST_QUERIES} --k=${K} --threads=${THREADS} --nodes=${NODES_REQ} --ppn=${PPN_REQ} 2>&1 | tee -a ${RUN_LOG}
        if [ -f "${PERF_DATA}" ]; then
            perf report --stdio -i ${PERF_DATA} > ${PERF_REPORT} 2>&1
            cp ${PERF_DATA} ${LATEST_PERF_DATA}
            cp ${PERF_REPORT} ${LATEST_PERF_REPORT}
            {
                echo "perf_data: ${PERF_DATA}"
                echo "latest_perf_data: ${LATEST_PERF_DATA}"
                echo "perf_report: ${PERF_REPORT}"
                echo "latest_perf_report: ${LATEST_PERF_REPORT}"
                echo "perf_report_top:"
                sed -n "1,${PERF_REPORT_LINES}p" ${PERF_REPORT}
            } | tee -a ${RUN_LOG}
        else
            echo "warning: perf data was not generated: ${PERF_DATA}" | tee -a ${RUN_LOG}
        fi
        ;;
    *)
        echo "warning: unknown PERF=${PERF}; running without perf" | tee ${RUN_LOG}
        /usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/main --algo=${ALGO} --nlist=${NLIST} --nprobe=${NPROBE} --test=${TEST_QUERIES} --k=${K} --threads=${THREADS} --nodes=${NODES_REQ} --ppn=${PPN_REQ} | tee -a ${RUN_LOG}
        ;;
esac

case "${PERF}" in
    1|stat|record)
        cp ${RUN_LOG} ${LATEST_RUN_LOG}
        ;;
esac

HEADER=$(grep '^csv_header: ' ${RUN_LOG} | tail -n 1 | sed 's/^csv_header: //')
CSV_LINE=$(grep '^csv: ' ${RUN_LOG} | tail -n 1 | sed 's/^csv: //')

if [ -n "${CSV_LINE}" ]; then
    ssh master_ubss1 "mkdir -p ${MASTER_RESULT_DIR}; lock=${MASTER_RESULT_DIR}/.ivf_mpi_results.lock; while ! mkdir \"\$lock\" 2>/dev/null; do sleep 1; done; trap 'rmdir \"\$lock\"' EXIT; if [ ! -s ${MASTER_RESULT_CSV} ] && [ -n '${HEADER}' ]; then printf '%s\n' '${HEADER}' > ${MASTER_RESULT_CSV}; fi; printf '%s\n' '${CSV_LINE}' >> ${MASTER_RESULT_CSV}" 2>&1
    echo "compact_csv: ${CSV_LINE}"
    echo "result_csv: ${MASTER_RESULT_CSV}"
else
    echo "warning: no csv line found in ${RUN_LOG}; check test.o and test.e"
fi

ssh master_ubss1 "mkdir -p ${MASTER_RESULT_DIR}" 1>&2
scp ${RUN_LOG} master_ubss1:${MASTER_RESULT_DIR}/ 2>&1
scp ${RESULT_DIR}/ivf_mpi_${PBS_JOBID}_perf* master_ubss1:${MASTER_RESULT_DIR}/ 2>/dev/null || true

for node in $NODES; do
    scp ${node}:/home/${USER}/files/ivf_centroids_*.bin master_ubss1:${MASTER_RESULT_DIR}/ 2>/dev/null || true
    scp ${node}:/home/${USER}/files/ivf_mpi_local_*.bin master_ubss1:${MASTER_RESULT_DIR}/ 2>/dev/null || true
done
