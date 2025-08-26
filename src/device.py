import os
import jax

print("------------------------------ Check for devices ------------------------")
print("--- Device Information ---")

# --- Detect GPUs/Accelerators using JAX ---
# JAX automatically detects and manages available accelerators (GPUs, TPUs).
# You can query the number of devices JAX sees.
class device_check:
    def __init__(self):
        pass
    def check(self):
        try:
            # Get all available devices (includes CPU, GPU, TPU)
            devices = jax.devices()
            num_devices = len(devices)
            print(f"Total JAX devices detected: {num_devices}")

            # Filter for specific device types (e.g., GPU)
            gpu_devices = [d for d in devices if d.device_kind == 'gpu']
            num_gpus = len(gpu_devices)
            print(f"Number of GPUs detected by JAX: {num_gpus}")

            if num_gpus > 0:
                print("GPU devices details:")
                for i, gpu in enumerate(gpu_devices):
                    print(f"  GPU {i}: {gpu}")
            else:
                print("No GPUs detected by JAX. Running on CPU or other default device.")

            # You can also check if JAX is configured to use a GPU
            if jax.default_backend() == 'gpu':
                print("JAX default backend is set to GPU.")
            elif jax.default_backend() == 'cpu':
                print("JAX default backend is set to CPU.")
            else:
                print(f"JAX default backend is set to: {jax.default_backend()}")

        except Exception as e:
            print(f"Could not query JAX devices: {e}")
            print("Ensure JAX and jaxlib with appropriate backend (e.g., jaxlib-cuda) are correctly installed.")

        print("\n--- CPU Information ---")

        # --- Detect CPU Cores ---
        # The 'os' module can provide information about the CPU.
        # os.cpu_count() returns the number of logical CPUs in the system.
        try:
            num_cpus = os.cpu_count()
            print(f"Number of logical CPU cores detected: {num_cpus}")

            # For more detailed CPU info (Linux specific, often from /proc/cpuinfo)
            # This is not always necessary but can be useful for debugging.
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = f.read()
                    processors = cpu_info.count('processor\t:')
                    print(f"Number of physical CPU processors (from /proc/cpuinfo): {processors}")

        except Exception as e:
            print(f"Could not query CPU information: {e}")

        print("\n--- JAX Device Placement (for your MCMC run) ---")
        print("NumPyro and JAX are designed to automatically use the fastest available device (GPU if detected).")
        print("You typically don't need to explicitly tell NumPyro/JAX to use the GPU if jaxlib-cuda is installed.")
        print("For specific device placement or multi-GPU setups, refer to JAX/NumPyro documentation on pmap/pjit.")
