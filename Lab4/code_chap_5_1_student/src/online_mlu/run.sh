rm *.pyc
rm *.jpg
python power_difference_test_bcl.py
python transform_mlu.py ../../images/chicago.jpg ../../models/int_pb_models/udnie_int8.pb  ../../models/int_pb_models/udnie_int8_power_diff.pb ../../models/int_pb_models/udnie_int8_power_diff_numpy.pb
