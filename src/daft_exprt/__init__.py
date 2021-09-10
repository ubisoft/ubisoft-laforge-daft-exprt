import os
import platform
import subprocess


# check platform
if platform.system() == "Linux":
    # REAPER binary
    binary_dir = os.path.join(os.path.dirname(__file__), 'bin', 'reaper', 'linux')
    os.environ['PATH'] += os.pathsep + binary_dir
    # binary requires minimum version for glibc
    ldd_version = subprocess.check_output("ldd --version | awk '/ldd/{print $NF}'", shell=True)
    ldd_version = float(ldd_version.decode('utf-8').strip())
    if ldd_version < 2.29:
        raise Exception(f'REAPER binary -- Unsupported ldd version: {ldd_version} < 2.29')
    # make binary executable for all groups
    binary_file = os.path.join(binary_dir, 'reaper')
    os.chmod(binary_file, 0o0755)
else:
    raise Exception(f'Unsupported platform: {os.platform.system()}')
