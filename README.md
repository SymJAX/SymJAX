<div align="center">
<img src="https://raw.githubusercontent.com/RandallBalestriero/SymJAX/master/doc/img/logo.png" alt="logo"></img>
</div>

# SymJAX: symbolic CPU/GPU/TPU programming
conda list --explicit > spec-file.txt
conda create --name myenv --file spec-file.txt

    pip freeze>requirements.txt


    pip download -r requirements.txt -d path_to_the_folder


    pip install -r requirements.txt --find-links=path_to_the_folder


export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH 
export LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LIBRARY_PATH                
export DATASET_PATH='XXX'                                    
export XLA_PYTHON_CLIENT_PREALLOCATE='false'                                export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-10.1"             export CUDA_DIR="/usr/local/cuda-10.1"
