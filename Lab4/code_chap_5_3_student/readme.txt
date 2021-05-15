实验步骤：
1.环境变量初始化：先进入env/，执行source env.sh; 再进入tensorflow-v1.10/, 执行source env.sh
2.bangpy算子填写：进入bangpy/PluginPowerDifferenceOp/目录，补全powerDIffBangpy.py, plugin_power_difference_kernel.h 和 powerDiff.cpp，执行make.sh进行编译，运行power_diff_test测试
3.重复实验5-1的操作，将bangpy版本的powerdifference算子集成进cnplugin以及tensorflow，完成在线和离线推理。
在已经5-1实验已经完成了的情况下，可以直接将新编译的libcnplugin.so拷入到env/neuware/lib64目录下，即可直接进行在线推理，不需要重新编译tensorflow

自动测试需要提交的文件：
├── inference.cpp
├── libcnplugin.so      // 重新编译cnplugin生成的库文件
├── plugin_power_difference_kernel.h
├── powerDIffBangpy.py
├── powerDiff.cpp
├── power_difference_test_bcl.py
├── tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl      // 重新编译tensorflow生成的whl
└── transform_mlu.py
将以上文件压缩成zip格式进行提交