"""
Minimal torchrun-equivalent launcher for Intel Gaudi (HPU).

Uses optimum-habana's DistributedRunner when available,
falls back to torch.distributed.run for CPU/testing.

Usage:
    python gaudi_spawn.py --nproc_per_node 4 train_sft_gaudi.py [script args...]
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--master_port", type=int, default=29500)
    known, rest = p.parse_known_args()

    if not rest:
        print("Usage: python gaudi_spawn.py --nproc_per_node N script.py [args...]")
        sys.exit(1)

    try:
        from optimum.habana.distributed import DistributedRunner
        runner = DistributedRunner(
            command_list=rest,
            world_size=known.nproc_per_node,
            use_mpi=False,
        )
        ret = runner.run()
        sys.exit(ret)
    except ImportError:
        # Fallback to torchrun (works for testing without Gaudi)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={known.nproc_per_node}",
            f"--master_port={known.master_port}",
        ] + rest
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
