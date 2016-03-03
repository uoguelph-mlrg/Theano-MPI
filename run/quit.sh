screen -X -S server quit
screen -X -S worker0 quit
screen -X -S worker1 quit
screen -X -S worker2 quit
screen -X -S worker3 quit

pkill -9 -f mpirun
