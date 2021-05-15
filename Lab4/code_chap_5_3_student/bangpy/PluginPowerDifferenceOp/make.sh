python3 powerDIffBangpy.py
cncc -c --bang-mlu-arch=MLU200 plugin_power_difference_kernel.mlu -o powerdiffkernel.o
g++ -c main.cpp
g++ -c powerDiff.cpp -I$NEUWARE_HOME/include
g++ powerdiffkernel.o main.o powerDiff.o -o power_diff_test -L $NEUWARE_HOME/lib64 -lcnrt

