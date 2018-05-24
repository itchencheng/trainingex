#!/usr/bin/env sh
set -e

TOOLS=$CAFFE_ROOT/build/tools

MYDATE=`date +%y%m%d-%H%M%S`
LOGDIR=log/
LOGFILE=${LOGDIR}log_$MYDATE.txt

$TOOLS/caffe train \
  --solver=cifar10_full_solver.prototxt $@ 2>&1 | tee $LOGFILE

# $TOOLS/caffe train \
#  --solver=examples/cifar10/cifar10_quick_solver.prototxt $@ 2>&1 | tee $LOGFILE


# reduce learning rate by factor of 10 after 8 epochs
# $TOOLS/caffe train \
#  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
#  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate $@ 2>&1 | tee $LOGFILE

