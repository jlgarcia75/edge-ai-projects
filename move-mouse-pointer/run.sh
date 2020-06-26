source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
PRECISION=$1
DEVICE=$2
python main.py -i ./resources/demo.mp4 -d ${DEVICE} -p ${PRECISION} -nf 5
