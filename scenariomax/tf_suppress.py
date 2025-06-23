"""TensorFlow suppression module - import this before any TensorFlow imports."""

import os
import warnings


# Set environment variables before any TensorFlow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all TF C++ logs (0=INFO, 1=WARN, 2=ERROR, 3=FATAL)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations messages
os.environ["TF_AUTOTUNE_THRESHOLD"] = "2"  # Reduce autotune verbosity

# CUDA-specific suppressions
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA completely (if you don't need GPU)
# OR if you need GPU, use these instead:
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# XLA and computation placer suppressions
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="

# Suppress specific library warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Additional TensorFlow internal suppressions
os.environ["TF_ENABLE_MKLDNN"] = "0"
os.environ["TF_ENABLE_GPU"] = "0"  # Disable GPU entirely if not needed

# Suppress Python warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore")

# Additional logging suppressions
import logging  # noqa: E402


logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("absl").setLevel(logging.FATAL)

# Import TensorFlow after setting environment variables
try:
    import tensorflow as tf

    # Configure TensorFlow logging
    tf.get_logger().setLevel("FATAL")
    tf.autograph.set_verbosity(0)

    # Disable eager execution warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # Additional TF suppressions
    tf.config.experimental.enable_tensor_float_32_execution(False)

except ImportError:
    pass  # TensorFlow not installed
