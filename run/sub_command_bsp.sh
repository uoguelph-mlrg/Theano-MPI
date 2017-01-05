# source ./set4theano-new.sh
num_worker=$1
unit='g'
mem=$((num_worker*10))$unit
ncpu=$((num_worker*2))

sqsub -q gpu -f mpi -n $ncpu -r 7d -o %J.out --gpp=$num_worker --mpp=$mem --nompirun ./run_bsp_workers.sh $num_worker