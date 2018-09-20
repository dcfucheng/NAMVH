TOOLS=./build/tools
LOG=examples/mnist/Log/my.log
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt --gpu=0 2>&1 | tee $LOG
