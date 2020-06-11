source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
PRECISION=$1
DEVICE=$2
python main.py -i ../bin/demo.mp4 \
	-fdm ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
	-flm ../models/intel/landmarks-regression-retail-0009 \
	-hpm ../models/intel/head-pose-estimation-adas-0001 \
	-gem ../models/intel/gaze-estimation-adas-0002 \
	-d ${DEVICE} \
	-bm True \
	-sv True \
	-ct 0.3 \
	-p ${PRECISION} \
	-nf 100
