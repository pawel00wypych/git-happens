FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    gnuradio-dev \
    gr-osmosdr \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libarmadillo-dev \
    libfftw3-dev \
    libvolk2-dev \
    libblas-dev \
    liblapack-dev \
    libmatio-dev \
    libpcap-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libpugixml-dev \
    libssl-dev \
    libuhd-dev \
    libgtest-dev \
    python3-mako \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/gnss-sdr/gnss-sdr.git

WORKDIR /app/gnss-sdr

RUN mkdir -p build

WORKDIR /app/gnss-sdr/build

RUN cmake .. && make -j$(nproc)

RUN make install

CMD ["gnss-sdr", "--help"]