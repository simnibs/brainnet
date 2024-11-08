from pathlib import Path

def process_dir(d, root_dest=None):
    # root = d if root is None else root

    # print(f"looking in {d}")

    for f in sorted(d.glob("*")):
        if f.is_dir():
            ds, sub, remainder = f.name.split(".")
            assert remainder == "surf_dir"
            process_dir(f, root_dest=d / ds / sub)
        else:
            if root_dest is not None:
                remainder = f.name
                dest = root_dest
            else:
                ds, sub, *remainder = f.name.split(".")
                remainder = '.'.join(remainder)
                dest = d / ds / sub

            if not dest.exists():
                dest.mkdir(parents=True)
            target = f.rename(dest / remainder)
            print(target)
            # print(f"source          {f}")
            # print(f"destination     {dest / remainder}")
            # print()

def remove_empty_surf_dir(d):
    for f in sorted(d.glob("*")):
        if f.name.endswith("surf_dir"):
            f.rmdir()

d = Path("/mnt/projects/CORTECH/nobackup/training_data")

process_dir(d)

remove_empty_surf_dir(d)