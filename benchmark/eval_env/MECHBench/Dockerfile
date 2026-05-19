FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libasan6 \
    libubsan1 \
    gfortran \
    cmake \
    perl \
    python3 \
    python-is-python3 \
    python3-pip \
    git-lfs \
    openmpi-bin \
    wget \
    libopenmpi-dev \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 \
    libfltk1.3-dev \
    libfreetype6-dev  \
    libgl1-mesa-dev \
    libocct-foundation-dev \
    libocct-data-exchange-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.gz && \
    tar -xzf openmpi-4.1.2.tar.gz && \
    cd /app/openmpi-4.1.2 && \
    ./configure --prefix=/opt/openmpi && \
    make -j$(nproc) all && \
    make install && \
    rm -rf /app/openmpi-4.1.2 openmpi-4.1.2.tar.gz


RUN git lfs install && \
    git clone --branch latest-20260120 --depth 1 https://github.com/OpenRadioss/OpenRadioss.git && \
    mkdir /app/OpenRadioss/exec && \
    cd /app/OpenRadioss/starter && \
    chmod +x ./build_script.sh && \
    ./build_script.sh -arch=linux64_gf -release && \
    cp ./cbuild_starter_linux64_gf/starter_linux64_gf /app/OpenRadioss/exec/ && \
    cd /app/OpenRadioss/engine && \
    chmod +x ./build_script.sh && \
    ./build_script.sh -arch=linux64_gf -mpi=ompi -release && \
    ./build_script.sh -arch=linux64_gf -release && \
    cp ./cbuild_engine_linux64_gf_ompi/engine_linux64_gf_ompi /app/OpenRadioss/exec/ && \
    cp ./cbuild_engine_linux64_gf/engine_linux64_gf /app/OpenRadioss/exec/ && \
    rm -rf .git .gitattributes .gitignore .github Apptainer CMake_Compilers common_source Compiling_tools doc engine exec_escape qa-tests reader scripts starter template tools && \
    rm -rf ./*.md ./*.txt ./*.json ./*.bat

ENV OPENRADIOSS_PATH=/app/OpenRadioss
ENV RAD_CFG_PATH=$OPENRADIOSS_PATH/hm_cfg_files
ENV RAD_H3D_PATH=$OPENRADIOSS_PATH/extlib/h3d/lib/linux64
ENV OMP_STACKSIZE=400m
ENV OMP_NUM_THREADS=1
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$OPENRADIOSS_PATH/extlib/hm_reader/linux64/
ENV PATH=/opt/openmpi/bin:$PATH

RUN git clone https://github.com/OpenRadioss/Tools.git && \
    cd /app/Tools && \
    git checkout 7eb4f791303a4ffde50446e4d54dc89d5739e45e && \
    cd /app/Tools/output_converters/anim_to_vtk/linux64 && \
    chmod +x ./build.bash && \
    ./build.bash  && \
    cd /app/Tools/output_converters/th_to_csv/linux64 && \
    chmod +x ./build.bash && \
    ./build.bash && \
    cp -r /app/Tools/exec/* /app/OpenRadioss/exec/ && \
    rm -rf /app/Tools

RUN git clone https://github.com/BayesOptApp/MECHBench.git && \
    cd /app/MECHBench && \
    git checkout 9c1580165c8fc9b0cdc81df68acb125870e75eb0 && \
    cd /app/MECHBench/src/sob/physical_models && \
    mkdir -p lib && \
    rm -rf figures ML_Benchmarks results ./main*

COPY requirements.txt /app/MECHBench/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/MECHBench/requirements.txt

COPY server.py /app/server.py
COPY /lib_mechbench /app/MECHBench/src/sob/physical_models/lib/
COPY /simulator_variable_bounds /app/simulator_variable_bounds/

# Create a non-root user
RUN useradd -m -s /bin/bash mechuser && \
    chown -R mechuser:mechuser /app && \
    chown -R mechuser:mechuser /opt

# Switch to the new user
USER mechuser

#HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#    CMD ["curl", "-f", "http://localhost:8000/health"]

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
