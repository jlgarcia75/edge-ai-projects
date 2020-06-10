
(base) c:\Users\jlgarci2\Dropbox\github\move-mouse-pointer\src>python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d CPU -ct 0.3 -async False -sv False
Video being shown:  False
Beginning test for precision FP32.
Completed run for precision FP32.
Beginning test for precision FP16.
Completed run for precision FP16.
OpenVINO Results
Current date and time:  2020-06-10 09:26:59
Platform: win32
Device: CPU
Asynchronous Inference: False
Precision: FP32,FP16
Total frames: 100
Total runtimes(s):
      Total runtime       FPS
FP32     123.695371  0.808438
FP16     123.438669  0.810119

Total Durations(ms) per phase:
                                  load  |   input   |   infer  | output
Model              Precision
facial detection  | FP32|477.3187  110.8960  1578.3580   8.7551
                  | FP16 |      253.2533  111.4063  1591.1398   9.6300
landmark detection| FP32  |     166.1206   10.0438    64.3487   7.2851
                  | FP16   |    118.2736   10.1416    65.0417   7.4530
head pose         | FP32    |   125.2244    8.9786   145.4375  12.5562
                  | FP16     |  137.0907    9.0327   143.3903  11.9578
gaze estimation   | FP32      | 127.2796    7.5069   154.1379   4.8336
                  | FP16       | 139.3510    4.4542   154.1262   4.9598

Duration(ms)/Frames per phase:
                                  load     input      infer    output
Model              Precision
facial detection   FP32       4.773187  1.108960  15.783580  0.087551
                   FP16       2.532533  1.114063  15.911398  0.096300
landmark detection FP32       1.661206  0.100438   0.643487  0.072851
                   FP16       1.182736  0.101416   0.650417  0.074530
head pose          FP32       1.252244  0.089786   1.454375  0.125562
                   FP16       1.370907  0.090327   1.433903  0.119578
gaze estimation    FP32       1.272796  0.075069   1.541379  0.048336
                   FP16       1.393510  0.044542   1.541262  0.049598

*********************************************************************************




(base) c:\Users\jlgarci2\Dropbox\github\move-mouse-pointer\src>python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d CPU -ct 0.3 -async True -sv False
Video being shown:  False
Beginning test for precision FP32.
Completed run for precision FP32.
Beginning test for precision FP16.
Completed run for precision FP16.
OpenVINO Results
Current date and time:  2020-06-10 09:31:07
Platform: win32
Device: CPU
Asynchronous Inference: True
Precision: FP32,FP16
Total frames: 100
Total runtimes(s):
      Total runtime       FPS
FP32     123.135343  0.812115
FP16     123.447517  0.810061

Total Durations(ms) per phase:
                                  load     input      infer   output
Model              Precision
facial detection   FP32       276.5253  100.3509  1487.2646   8.8087
                   FP16       277.1853  117.7123  1661.4165   9.6197
landmark detection FP32       112.1027   44.4284     0.0000  44.9154
                   FP16       118.2556   47.7438     0.0000  49.5373
head pose          FP32       119.5457   42.4560     0.0000  53.3735
                   FP16       138.7391   42.9170     0.0000  54.2341
gaze estimation    FP32       123.9540    4.6744   161.2043   5.4730
                   FP16       142.2412    4.9023   163.7634   5.3270

Duration(ms)/Frames per phase:
                                  load     input      infer    output
Model              Precision
facial detection   FP32       2.765253  1.003509  14.872646  0.088087
                   FP16       2.771853  1.177123  16.614165  0.096197
landmark detection FP32       1.121027  0.444284   0.000000  0.449154
                   FP16       1.182556  0.477438   0.000000  0.495373
head pose          FP32       1.195457  0.424560   0.000000  0.533735
                   FP16       1.387391  0.429170   0.000000  0.542341
gaze estimation    FP32       1.239540  0.046744   1.612043  0.054730
                   FP16       1.422412  0.049023   1.637634  0.053270

*********************************************************************************




(base) c:\Users\jlgarci2\Dropbox\github\move-mouse-pointer\src>python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d HETERO:GPU,CPU -ct 0.3 -async False -sv False
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
mbox_conf_reshape/DimData_const is GPU
conv4_3_0_norm_mbox_loc_flat is GPU
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_flat is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_flat is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_flat is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_flat is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_flat is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_flat is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_flatten is CPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_flat is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_flat is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_flat is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_flat is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_flat is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_flat is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_flat is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_p_fc/flatten_fc_input/Cast_14129_const is GPU
angle_r_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input/Cast_14127_const is GPU
angle_y_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input/Cast_14125_const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input is GPU
130 is GPU
130/Cast_15117_const is GPU
138 is GPU
138/Cast_15115_const is GPU
140/Dims/Output_0/Data__const is GPU
141/Dims/Output_0/Data__const is GPU
150/Dims/Output_0/Data__const is GPU
gaze_vector/Dims/Output_0/Data__const is GPU
gaze_vector is GPU
150 is GPU
141250 is GPU
140256 is GPU
138 is GPU
130 is GPU
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
mbox_conf_reshape/DimData_const is GPU
conv4_3_0_norm_mbox_loc_flat is GPU
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_flat is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_flat is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_flat is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_flat is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_flat is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_flat is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_flatten is CPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_flat is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_flat is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_flat is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_flat is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_flat is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_flat is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_flat is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_p_fc/flatten_fc_input/Cast_14129_const is GPU
angle_r_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input/Cast_14127_const is GPU
angle_y_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input/Cast_14125_const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input is GPU
130 is GPU
130/Cast_15117_const is GPU
138 is GPU
138/Cast_15115_const is GPU
140/Dims/Output_0/Data__const is GPU
141/Dims/Output_0/Data__const is GPU
150/Dims/Output_0/Data__const is GPU
gaze_vector/Dims/Output_0/Data__const is GPU
gaze_vector is GPU
150 is GPU
141252 is GPU
140256 is GPU
138 is GPU
130 is GPU
Video being shown:  False
Beginning test for precision FP32.
I see you. Move the mouse cursor with your eyes.
Completed run for precision FP32.
Beginning test for precision FP16.
I see you. Move the mouse cursor with your eyes.
Completed run for precision FP16.
OpenVINO Results
Current date and time:  2020-06-10 09:36:27
Platform: win32
Device: HETERO:GPU,CPU
Asynchronous Inference: False
Precision: FP32,FP16
Total frames: 100
Total runtimes(s):
      Total runtime       FPS
FP32     160.078532  0.624693
FP16     158.048177  0.632718

Total Durations(ms) per phase:
                                    load     input      infer   output
Model              Precision
facial detection   FP32       22317.1596  116.8904  2090.5380  10.1817
                   FP16       20608.3234  137.5879  2111.7795  10.3625
landmark detection FP32        3691.4430   11.3018   180.6220   8.2680
                   FP16        3314.7605   11.6176   198.9666   8.1540
head pose          FP32        4991.0941   10.5762   257.7362  10.8117
                   FP16        5334.4964   10.5763   243.1082  11.1680
gaze estimation    FP32        5461.2131    5.2269   265.3656   5.1601
                   FP16        5397.8235    5.2020   243.4360   5.1853

Duration(ms)/Frames per phase:
                                    load     input      infer    output
Model              Precision
facial detection   FP32       223.171596  1.168904  20.905380  0.101817
                   FP16       206.083234  1.375879  21.117795  0.103625
landmark detection FP32        36.914430  0.113018   1.806220  0.082680
                   FP16        33.147605  0.116176   1.989666  0.081540
head pose          FP32        49.910941  0.105762   2.577362  0.108117
                   FP16        53.344964  0.105763   2.431082  0.111680
gaze estimation    FP32        54.612131  0.052269   2.653656  0.051601
                   FP16        53.978235  0.052020   2.434360  0.051853

*********************************************************************************




(base) c:\Users\jlgarci2\Dropbox\github\move-mouse-pointer\src>python main.py -nf 100 -p FP32,FP16 -i ..\bin\demo.mp4 -fdm ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -flm ..\models\intel\landmarks-regression-retail-0009 -hpm ..\models\intel\head-pose-estimation-adas-0001 -gem ..\models\intel\gaze-estimation-adas-0002 -d HETERO:GPU,CPU -ct 0.3 -async True -sv False
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
mbox_conf_reshape/DimData_const is GPU
conv4_3_0_norm_mbox_loc_flat is GPU
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_flat is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_flat is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_flat is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_flat is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_flat is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_flat is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_flatten is CPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_flat is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_flat is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_flat is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_flat is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_flat is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_flat is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_flat is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_p_fc/flatten_fc_input/Cast_14129_const is GPU
angle_r_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input/Cast_14127_const is GPU
angle_y_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input/Cast_14125_const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input is GPU
130 is GPU
130/Cast_15117_const is GPU
138 is GPU
138/Cast_15115_const is GPU
140/Dims/Output_0/Data__const is GPU
141/Dims/Output_0/Data__const is GPU
150/Dims/Output_0/Data__const is GPU
gaze_vector/Dims/Output_0/Data__const is GPU
gaze_vector is GPU
150 is GPU
141250 is GPU
140256 is GPU
138 is GPU
130 is GPU
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
mbox_conf_reshape/DimData_const is GPU
conv4_3_0_norm_mbox_loc_flat is GPU
conv4_3_0_norm_mbox_loc_perm is GPU
conv4_3_norm_mbox_loc_flat is GPU
conv4_3_norm_mbox_loc_perm is GPU
fc7_mbox_loc_flat is GPU
fc7_mbox_loc_perm is GPU
conv6_2_mbox_loc_flat is GPU
conv6_2_mbox_loc_perm is GPU
conv7_2_mbox_loc_flat is GPU
conv7_2_mbox_loc_perm is GPU
conv8_2_mbox_loc_flat is GPU
conv8_2_mbox_loc_perm is GPU
conv9_2_mbox_loc_flat is GPU
conv9_2_mbox_loc_perm is GPU
mbox_conf_flatten is CPU
mbox_conf_reshape is GPU
conv9_2_mbox_conf_flat is GPU
conv9_2_mbox_conf_perm is GPU
conv8_2_mbox_conf_flat is GPU
conv8_2_mbox_conf_perm is GPU
conv7_2_mbox_conf_flat is GPU
conv7_2_mbox_conf_perm is GPU
conv6_2_mbox_conf_flat is GPU
conv6_2_mbox_conf_perm is GPU
fc7_mbox_conf_flat is GPU
fc7_mbox_conf_perm is GPU
conv4_3_0_norm_mbox_conf_flat is GPU
conv4_3_0_norm_mbox_conf_perm is GPU
conv4_3_norm_mbox_conf_flat is GPU
conv4_3_norm_mbox_conf_perm is GPU
712_const is GPU
714_const is GPU
720_const is GPU
722_const is GPU
728_const is GPU
730_const is GPU
736_const is GPU
738_const is GPU
744_const is GPU
746_const is GPU
752_const is GPU
754_const is GPU
760_const is GPU
762_const is GPU
768_const is GPU
770_const is GPU
776_const is GPU
778_const is GPU
784_const is GPU
786_const is GPU
792_const is GPU
794_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1184_const is GPU
Copy_L0008_ActivationBin-back_bone_seq.conv2_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1182_const is GPU
Copy_L0013_ActivationBin-back_bone_seq.conv3_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1192_const is GPU
Copy_L0018_ActivationBin-back_bone_seq.conv3_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1174_const is GPU
Copy_L0023_ActivationBin-back_bone_seq.conv4_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1178_const is GPU
Copy_L0028_ActivationBin-back_bone_seq.conv4_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1172_const is GPU
Copy_L0033_ActivationBin-back_bone_seq.conv5_1_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1180_const is GPU
Copy_L0038_ActivationBin-back_bone_seq.conv5_2_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1176_const is GPU
Copy_L0043_ActivationBin-back_bone_seq.conv5_3_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1186_const is GPU
Copy_L0048_ActivationBin-back_bone_seq.conv5_4_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1188_const is GPU
Copy_L0053_ActivationBin-back_bone_seq.conv5_5_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data_1190_const is GPU
Copy_L0060_ActivationBin-back_bone_seq.conv5_6_sep_relubin_bin_conv_BIN01/Input_1/Output_0/Data__const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_p_fc/flatten_fc_input/Cast_14129_const is GPU
angle_r_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input/Cast_14127_const is GPU
angle_y_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input/Cast_14125_const is GPU
angle_p_fc/flatten_fc_input is GPU
angle_r_fc/flatten_fc_input is GPU
angle_y_fc/flatten_fc_input is GPU
130 is GPU
130/Cast_15117_const is GPU
138 is GPU
138/Cast_15115_const is GPU
140/Dims/Output_0/Data__const is GPU
141/Dims/Output_0/Data__const is GPU
150/Dims/Output_0/Data__const is GPU
gaze_vector/Dims/Output_0/Data__const is GPU
gaze_vector is GPU
150 is GPU
141252 is GPU
140256 is GPU
138 is GPU
130 is GPU
Video being shown:  False
Beginning test for precision FP32.
I see you. Move the mouse cursor with your eyes.
Completed run for precision FP32.
Beginning test for precision FP16.
I see you. Move the mouse cursor with your eyes.
Completed run for precision FP16.
OpenVINO Results
Current date and time:  2020-06-10 09:41:46
Platform: win32
Device: HETERO:GPU,CPU
Asynchronous Inference: True
Precision: FP32,FP16
Total frames: 100
Total runtimes(s):
      Total runtime       FPS
FP32     158.475282  0.631013
FP16     158.315022  0.631652

Total Durations(ms) per phase:
                                    load     input      infer   output
Model              Precision
facial detection   FP32       21326.6917  121.2836  2122.4262  10.1803
                   FP16       21020.6454  134.9714  2189.1834  10.2359
landmark detection FP32        3708.6516   52.7991     0.0000  63.4272
                   FP16        3294.3355   54.1614     0.0000  60.3203
head pose          FP32        4857.0317   44.0176     0.0000  68.0128
                   FP16        5308.5247   44.5865     0.0000  63.2842
gaze estimation    FP32        5310.4178    5.5606   264.9415   5.1436
                   FP16        5403.1593    5.2428   243.7323   4.7677

Duration(ms)/Frames per phase:
                                    load     input      infer    output
Model              Precision
facial detection   FP32       213.266917  1.212836  21.224262  0.101803
                   FP16       210.206454  1.349714  21.891834  0.102359
landmark detection FP32        37.086516  0.527991   0.000000  0.634272
                   FP16        32.943355  0.541614   0.000000  0.603203
head pose          FP32        48.570317  0.440176   0.000000  0.680128
                   FP16        53.085247  0.445865   0.000000  0.632842
gaze estimation    FP32        53.104178  0.055606   2.649415  0.051436
                   FP16        54.031593  0.052428   2.437323  0.047677

*********************************************************************************
