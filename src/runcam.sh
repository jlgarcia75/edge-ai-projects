source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
PRECISION=$1
DEVICE=$2
python main.py -i 0 \
	-fdm /Users/jesusgarcia/openvino_models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
	-flm /Users/jesusgarcia/openvino_models/intel/landmarks-regression-retail-0009/${PRECISION}/landmarks-regression-retail-0009 \
	-hpm /Users/jesusgarcia/openvino_models/intel/head-pose-estimation-adas-0001/${PRECISION}/head-pose-estimation-adas-0001 \
	-gem /Users/jesusgarcia/openvino_models/intel/gaze-estimation-adas-0002/${PRECISION}/gaze-estimation-adas-0002 \
	-d ${DEVICE} \
	-ct 0.3 \
	-p ${PRECISION} \
	-v True
