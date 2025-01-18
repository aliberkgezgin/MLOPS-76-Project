before = set(open("before_cleanup.txt").readlines())
after = set(open("after_cleanup.txt").readlines())
missing = before - after

with open("missing_packages.txt", "w") as f:
    f.writelines(sorted(missing))
print("Missing packages saved to missing_packages.txt")
