# AutoScope, the stock-OpenCV container.
#
# One image, two architectures. Build it for linux/arm64 and it is the
# "c8g stock" arm of the COOL comparison and the shape of what runs under AWS
# Batch; build it for linux/amd64 and it is the "c7i" arm. The COOL arm itself
# is NOT this file: COOL ships as an AMI with its own OpenCV build, and the
# container for that arm is derived from the AMI (infra/, to come). Keeping
# the stock control in a Dockerfile anybody can build is what makes the
# comparison checkable.
#
#   docker build -t autoscope .                                   # this machine
#   docker build --platform linux/arm64 -t autoscope:arm64 .      # Arm, under emulation on x86
#   docker run --rm autoscope                                     # runs the tests
#   docker run --rm autoscope python bench/cool_bench.py --runs 100 --label c8g-stock
#
# Timings taken under emulation mean nothing. Correctness does.

FROM python:3.12-slim

# opencv-python (the wheel we pin) links libGL and glib even when no window is
# ever opened. Two small packages, and the same wheel as on our laptops, so a
# number measured here is a number measured with the same binary.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download them.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY tests ./tests
COPY bench ./bench
# The eval scripts, not their results: tests/test_export_detection.py imports
# one, and the chain has to be runnable from the image on a Graviton box.
COPY eval/*.py eval/chain.sh ./eval/
RUN pip install --no-cache-dir --no-deps .

# The JPEG warning on a handful of NLM files is noise, not a failure.
ENV OPENCV_LOG_LEVEL=ERROR \
    PYTHONUNBUFFERED=1

# Data is never in the image. Mount it:  -v $PWD/data:/app/data
VOLUME ["/app/data"]

CMD ["pytest", "-q"]
