import os

# Test connection to Sonic HPC Cluster by printing details about the system

print(f"Current working directory: {os.getcwd()}")
print(f"Processor count: {os.cpu_count()}")
print(f"Memory info: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3):.2f} GB")
print(f"System platform: {os.name}")
print(f"Platform: {os.uname()}")