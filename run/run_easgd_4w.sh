
#### device 1
if [[ -z $1 ]]; then
	echo 'need a device1 as argument $1'
	exit 1
else
	device1=$1
fi

if [[ ${device1:0:3} == "gpu" ]]; then
	
	dev1=${device1#gpu}
else
	echo 'device1 starts with *gpu* '
	exit 1
fi

# if [[ $dev1 -ge '4' ]]; then
# 	numa1=1
# else
# 	numa1=0
# fi
#
# echo 'numa1:' $numa1 'device1:' $1

######### device 2

if [[ -z $2 ]]; then
	echo 'need a device2 as argument $1'
	exit 1
else
	device2=$2
fi

if [[ ${device2:0:3} == "gpu" ]]; then
	
	dev2=${device2#gpu}
else
	echo 'device2 starts with *gpu* '
	exit 1
fi

# if [[ $dev2 -ge '4' ]]; then
# 	numa2=1
# else
# 	numa2=0
# fi
#
# echo 'numa2:' $numa2 'device2:' $2

###### device 3

if [[ -z $3 ]]; then
	echo 'need a device3 as argument $1'
	exit 1
else
	device3=$3
fi

if [[ ${device3:0:3} == "gpu" ]]; then
	
	dev3=${device3#gpu}
else
	echo 'device3 starts with *gpu* '
	exit 1
fi

# if [[ $dev3 -ge '4' ]]; then
# 	numa3=1
# else
# 	numa3=0
# fi
#
# echo 'numa3:' $numa3 'device3:' $3

###### device 4

if [[ -z $4 ]]; then
	echo 'need a device4 as argument $1'
	exit 1
else
	device4=$4
fi

if [[ ${device4:0:3} == "gpu" ]]; then
	
	dev4=${device4#gpu}
else
	echo 'device4 starts with *gpu* '
	exit 1
fi

# if [[ $dev4 -ge '4' ]]; then
# 	numa4=1
# else
# 	numa4=0
# fi
#
# echo 'numa4:' $numa4 'device4:' $4


#!/bin/bash
screen -Sdm server sh -c "source ./set4theano.sh; ./run_server.sh; exec bash"
screen -Sdm workers sh -c "source ./set4theano.sh; ./run_workers.sh '$device1' '$device2' '$device3' '$device4'; exec bash"

	





