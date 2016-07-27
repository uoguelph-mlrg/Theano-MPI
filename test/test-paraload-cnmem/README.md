replace the `proc_load_mpi.py` and `iterator.py` in `Theano-MPI/lib/base/` with those in this folder and run `./run_bsp_worker.sh 1` to test the compatibility of cnmem with parallel loading.

If compatible, the output of `data` (loaded shared_x view from loading process) will be the same as `iter` (loadded shared_x view from training process).

Try running with and without lib.cnmem=1 set in run_bsp_wroker.sh to see the difference.