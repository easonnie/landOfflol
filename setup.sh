export CUDA_HOME=$SHARED_ROOT/cuda-7.5
export LD_LIBRARY_PATH=$SHARED_ROOT/cuda-7.5/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SHARED_ROOT/cudnn-4.0.7/lib64:$LD_LIBRARY_PATH

export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR_TMP
