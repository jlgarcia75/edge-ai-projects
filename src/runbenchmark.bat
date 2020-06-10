python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d CPU -ct 0.3 -async False -sv False

python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d CPU -ct 0.3 -async True -sv False

python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d HETERO:GPU,CPU -ct 0.3 -async False -sv False

python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d HETERO:GPU,CPU -ct 0.3 -async True -sv False
