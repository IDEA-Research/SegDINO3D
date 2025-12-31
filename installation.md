# Installation

1. Install Python and Torch

    ```shell
    conda create -n dino3d python=3.9.20
    conda activate dino3d
    conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

2. Install MinkowskiEngine
  
    ```shell
    conda install openblas-devel -c anaconda
    cp ~/anaconda3/envs/dino3d/lib/libopenblas.so* ~/anaconda3/envs/dino3d/lib/python3.9/site-packages/torch/lib/
    cd your/path/packages # your path to save packages
    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    ```

    Before compilation, we need to modify the source code to fix a bug refer to this [issue](https://github.com/NVIDIA/MinkowskiEngine/issues/601):

    - MinkowskiEngine/src/3rdparty/concurrent_unordered_map.cuh: Add `#include <thrust/execution_policy.h>`

    - MinkowskiEngine/src/convolution_kernel.cuh: Add `#include <thrust/execution_policy.h>`

    - MinkowskiEngine/src/coordinate_map_gpu.cu: Add `#include <thrust/unique.h>` and `#include <thrust/remove.h>`

    - MinkowskiEngine/src/spmm.cu: Add `#include <thrust/execution_policy.h>`, `#include <thrust/reduce.h>`, and `#include <thrust/sort.h>`

    ```shell
    find ${CONDA_PREFIX}/include -name "cblas.h"
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    ```

3. Install OpenMMLab projects

    ```shell
    pip install --no-deps mmengine==0.10.6 mmdet==3.3.0
    pip install ninja psutil

    # install mmcv
    pip install setuptools wheel packaging pyyaml # Install the dependencies required for compilation manually
    cd your/path/packages # your path to save packages
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    git checkout v2.1.0
    FORCE_CUDA=1 pip install -e . --no-build-isolation -v

    pip install --no-deps mmdet3d==1.4
    ```

4. Install torch-scatter

    ```shell
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html --no-deps
    ```

5. Install ScanNet superpoint segmentator if you need to prepare superpoint (optional)

    ```shell
    cd your/path/packages # your path to save packages
    git clone https://github.com/Karbo123/segmentator.git
    cd segmentator

    cd csrc && mkdir build && cd build
    cmake .. \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

    make && make install # after install, please do not delete this folder (as we only create a symbolic link)
    ```

6. Install remaining python packages

    ```shell
    pip install --no-deps \
        spconv-cu121==2.3.8 \
        addict==2.4.0 \
        yapf==0.43.0 \
        termcolor==2.5.0 \
        packaging==24.2 \
        numpy==2.0.1 \
        rich==13.4.2 \
        opencv-python==4.11.0.86 \
        pycocotools==2.0.10 \
        Shapely==1.8.5.post1 \
        scipy==1.13.1 \
        terminaltables==3.1.10 \
        numba==0.60.0 \
        llvmlite==0.43.0 \
        pccm==0.4.16 \
        ccimport==0.4.4 \
        pybind11==2.13.6 \
        ninja==1.11.1.3 \
        lark==1.2.2 \
        cumm-cu121==0.7.11 \
        pyquaternion==0.9.9 \
        lyft-dataset-sdk==0.0.8 \
        pandas==2.2.3 \
        python-dateutil==2.9.0.post0 \
        matplotlib==3.9.4 \
        pyparsing==3.2.1 \
        cycler==0.12.1 \
        kiwisolver==1.4.7 \
        scikit-learn==1.6.1 \
        joblib==1.4.2 \
        threadpoolctl==3.5.0 \
        cachetools==5.5.2 \
        nuscenes-devkit==1.1.11 \
        trimesh==4.6.2 \
        open3d==0.19.0 \
        plotly==6.0.0 \
        dash==2.18.2 \
        plyfile==1.1 \
        flask==3.0.3 \
        werkzeug==3.0.6 \
        click==8.1.8 \
        blinker==1.9.0 \
        itsdangerous==2.2.0 \
        importlib_metadata==8.5.0 \
        zipp==3.21.0 \
        tqdm==4.67.1 \
        transformers==4.48.0 \
        huggingface-hub==0.32.4 \
        regex==2024.11.6 \
        safetensors==0.5.3 \
        tokenizers==0.21.1 \
        portalocker==3.1.1 \
        fire==0.7.0 \
        pytz==2023.4
    ```

7. Modify the source code in mmdet3d

- Find the file: `~/anaconda3/envs/dino3d/lib/python3.9/site-packages/mmdet3d/models/layers/spconv/overwrite_spconv/write_spconv2.py`

- Change the line `version = local metadata.get('version', None)` to `version = local metadata.get('version', 2)`
