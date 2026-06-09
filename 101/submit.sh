#!/bin/sh
set -eu

NODES_REQ=${1:-1}
PPN_REQ=${2:-8}
NP=${3:-8}
NLIST=${4:-512}
NPROBE=${5:-64}
TEST_QUERIES=${6:-2000}
K=${7:-10}
THREADS=1
ALGO=ivf-mpi
PERF=${PERF:-0}
PERF_SET=${PERF_SET:-arm}
PERF_REPEAT=${PERF_REPEAT:-3}
PERF_FREQ=${PERF_FREQ:-99}
PERF_REPORT_LINES=${PERF_REPORT_LINES:-80}

if [ "${PERF}" = "record" ]; then
    CXXFLAGS="-O2 -g -fno-omit-frame-pointer -std=c++11"
else
    CXXFLAGS="-O2 -std=c++11"
fi

echo "compile: mpic++ ${CXXFLAGS} main.cc -o main"
if ! grep -q 'int main' main.cc; then
    echo "error: current main.cc does not contain int main; copy the MPI main.cc into ~/ann first" >&2
    exit 1
fi
mpic++ ${CXXFLAGS} main.cc -o main

echo "submit: nodes=${NODES_REQ} ppn=${PPN_REQ} np=${NP} nlist=${NLIST} nprobe=${NPROBE} test=${TEST_QUERIES} k=${K}"
JOB_ID=$(qsub -l nodes=${NODES_REQ}:ppn=${PPN_REQ} -v NP=${NP},NLIST=${NLIST},NPROBE=${NPROBE},TEST_QUERIES=${TEST_QUERIES},K=${K},THREADS=${THREADS},NODES_REQ=${NODES_REQ},PPN_REQ=${PPN_REQ},ALGO=${ALGO},PERF=${PERF},PERF_SET=${PERF_SET},PERF_REPEAT=${PERF_REPEAT},PERF_FREQ=${PERF_FREQ},PERF_REPORT_LINES=${PERF_REPORT_LINES} qsub_mpi.sh)
echo "${JOB_ID}"

if [ "${WAIT_FOR_RESULT:-1}" = "0" ]; then
    echo "submitted without waiting."
    echo "check later: qstat ${JOB_ID}"
    echo "show latest rows: tail -n 5 files/ivf_mpi_results.csv"
    echo "show all rows: cat files/ivf_mpi_results.csv"
    exit 0
fi

echo "waiting for ${JOB_ID} to finish..."
WAIT_SECONDS=0
STATUS_EVERY=${STATUS_EVERY:-30}
MAX_QUEUE_WAIT=${MAX_QUEUE_WAIT:-300}
while qstat "${JOB_ID}" >/dev/null 2>&1; do
    JOB_STATE=$(qstat "${JOB_ID}" 2>/dev/null | tail -n 1 | awk '{print $(NF-1)}')
    if [ "${JOB_STATE}" = "C" ] || [ "${JOB_STATE}" = "E" ]; then
        echo "job state is ${JOB_STATE}; PBS has finished running it."
        break
    fi
    if [ "${JOB_STATE}" = "Q" ] && [ "${MAX_QUEUE_WAIT}" -gt 0 ] && [ "${WAIT_SECONDS}" -ge "${MAX_QUEUE_WAIT}" ]; then
        echo "job is still queued after ${WAIT_SECONDS}s; leaving it in PBS queue."
        echo "check later: qstat ${JOB_ID}"
        echo "cancel it: qdel ${JOB_ID}"
        echo "show latest rows after completion: tail -n 5 files/ivf_mpi_results.csv"
        echo "show all rows after completion: cat files/ivf_mpi_results.csv"
        exit 0
    fi
    if [ $((WAIT_SECONDS % STATUS_EVERY)) -eq 0 ]; then
        echo "still running after ${WAIT_SECONDS}s:"
        qstat "${JOB_ID}" 2>/dev/null | tail -n 1 || true
    fi
    sleep 5
    WAIT_SECONDS=$((WAIT_SECONDS + 5))
done

# PBS may flush stdout a moment after qstat stops listing the job.
sleep 2

echo "job finished: ${JOB_ID}"
SUMMARY_FILE=files/ivf_mpi_results.csv

if [ -f "${SUMMARY_FILE}" ]; then
    CSV_HEADER=$(head -n 1 "${SUMMARY_FILE}")
    CSV_ROW=$(tail -n 1 "${SUMMARY_FILE}")

    echo "latest experiment result:"
    awk -v header="${CSV_HEADER}" -v row="${CSV_ROW}" '
    BEGIN {
        n = split(header, keys, ",");
        split(row, vals, ",");
        for (i = 1; i <= n; ++i) {
            data[keys[i]] = vals[i];
        }

        print "config: algorithm=" data["algorithm"] \
              " np=" data["np"] \
              " threads=" data["threads"] \
              " nodes=" data["nodes"] \
              " ppn=" data["ppn"] \
              " nlist=" data["nlist"] \
              " nprobe=" data["nprobe"] \
              " k=" data["k"] \
              " test=" data["test_queries"] \
              " partition=" data["partition"] \
              " cache_hit=" data["cache_hit"];
        print "average recall: " data["recall"];
        print "average latency (us): " data["latency_us"];
        print "batch_time_ms: " data["batch_time_ms"];
        print "qps: " data["qps"];
        print "comm_us: bcast=" data["bcast_us"] \
              " gather=" data["gather_us"] \
              " merge=" data["merge_us"];
        print "search_us: avg=" data["search_avg_us"] \
              " max=" data["search_max_us"];
        print "work: candidates_avg=" data["candidate_count_avg"] \
              " items_min=" data["work_min"] \
              " items_avg=" data["work_avg"] \
              " items_max=" data["work_max"] \
              " imbalance=" data["imbalance_ratio"];
    }'

    CENTROID_BUILD=$(grep '^centroid_build_time_ms: ' test.o 2>/dev/null | tail -n 1 | sed 's/^centroid_build_time_ms: //')
    LOCAL_CACHE_HITS=$(grep '^local_index_cache_hits: ' test.o 2>/dev/null | tail -n 1 | sed 's/^local_index_cache_hits: //')
    LOCAL_BUILD_AVG=$(grep '^local_index_build_avg_ms: ' test.o 2>/dev/null | tail -n 1 | sed 's/^local_index_build_avg_ms: //')
    LOCAL_BUILD_MAX=$(grep '^local_index_build_max_ms: ' test.o 2>/dev/null | tail -n 1 | sed 's/^local_index_build_max_ms: //')
    echo "build/cache: centroid_ms=${CENTROID_BUILD:-NA} local_cache_hits=${LOCAL_CACHE_HITS:-NA} local_build_avg_ms=${LOCAL_BUILD_AVG:-NA} local_build_max_ms=${LOCAL_BUILD_MAX:-NA}"
    echo "csv file: ${SUMMARY_FILE}"
    TOTAL_LINES=$(wc -l < "${SUMMARY_FILE}")
    TOTAL_RESULTS=$((TOTAL_LINES - 1))
    echo "csv result rows: ${TOTAL_RESULTS}"
    echo "show all rows: cat ${SUMMARY_FILE}"
else
    echo "${SUMMARY_FILE} not found; check test.o and test.e"
fi
