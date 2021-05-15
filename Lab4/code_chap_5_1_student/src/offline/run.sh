export CNRT_GET_HARDWARE_TIME=off
export CNRT_PRINT_INFO=off
rm *.jpg
./bin/style_transfer chicago udnie_int8
./bin/style_transfer chicago udnie_int8_power_diff
