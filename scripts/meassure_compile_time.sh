nruns=$2
prog=$1
run() {
    make  ${prog}
}


for i in $(seq ${nruns}); do
    make clean
    time run
done
