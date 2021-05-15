实验步骤：
1、环境变量初始化：先进入env/，执行source env.sh; 再进入tensorflow-v1.10/,执行source env.sh
2、bangc算子填写：补齐src/bangc/PluginPowerDifferenceOp/plugin_power_difference_kernel.h,plgin_power_difference_kernel.mlu和powerDiff.cpp，执行make.sh进行编译，运行power_diff_test测试
3、集成到cnplugin: 补齐src/bangc/PluginPowerDifferenceOp/cnplugin.h和plugin_power_difference_op.cc，将整个PluginPowerDifferenceOp文件夹复制到env/Cambricon-CNPlugin-MLU270/pluginops,在Cambricon-CNPlugin-MLU270目录下执行build_cnplugin.sh重新编译cnplugin;编译完成后将build/libcnplugin.so和cnplugin.h分别拷入到env/neuware/lib64和env/neuware/include中。
 注：cnplugin.h中PowerDifference算子的声明可以参考其他算子声明来进行添加，plugin_power_difference_op.cc中的算子函数定义可以参考pluginops目录下其他算子的定义实现
4、集成到tensorflow: 补齐src/tf-implementation/tf-add-power-diff/power_difference.cc和cwise_op_power_difference_mlu.h，按照readme.txt提示拷入到对应文件夹，重新编译tensorflow
5、框架算子测试：补齐src/online_mlu/power_difference_test_bcl.py
6、在线推理和生成离线模型：补齐src/online_mlu/power_difference_test_bcl.pysrc/online_mlu/transform_mlu.py
7、离线推理：补齐src/offline/src/inference.cpp

自动测试需要提交的文件：
├── inference.cpp
├── libcnplugin.so      // 重新编译cnplugin生成的库文件
├── plugin_power_difference_kernel.h
├── plugin_power_difference_kernel.mlu
├── powerDiff.cpp
├── power_difference_test_bcl.py
├── tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl      // 重新编译tensorflow生成的whl
└── transform_mlu.py
将以上文件压缩成zip格式进行提交
