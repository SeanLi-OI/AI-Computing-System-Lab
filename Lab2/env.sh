#!/bin/bash

UNAME_V=`cat /etc/issue | head -n 1`

let INDEX=`expr "$UNAME_V" : ".*Debian.*"`
if [ $INDEX -gt 0 ]; then
    OS_VERSION="ubuntu16.04"
fi

let INDEX=`expr "$UNAME_V" : ".*Ubuntu.*"`
if [ $INDEX -gt 0 ]; then
    OS_VERSION="ubuntu16.04"
fi

if [ -f /etc/centos-release ]; then
    OS_VERSION="centos7.4"
    INSTALL_METHOD="yum install -y *"
fi

export NEUWARE_HOME=/usr/local/neuware
export PYTHONPATH=$PYTHONPATH
export ATEN_CNML_COREVERSION="MLU200"
export mcore="MLU200"
export PATH=$PATH:$NEUWARE_HOME/bin
export LD_LIBRARY_PATH=$NEUWARE_HOME/lib64
export GLOG_alsologtostderr=true
# Set log level which is output to stderr, 0: INFO/WARNING/ERROR/FATAL, 1: WARNING/ERROR/FATAL, 2: ERROR/FATAL, 3: FATAL,
export GLOG_minloglevel=0
