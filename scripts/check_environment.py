"""
check_environment.py – Verify all dependencies are installed and importable.
Run: python scripts/check_environment.py
"""
import sys
import importlib

REQUIRED = [
    ("torch",           "2.0.0"),
    ("torchvision",     "0.15.0"),
    ("streamlit",       "1.32.0"),
    ("pandas",          "2.0.0"),
    ("numpy",           "1.24.0"),
    ("PIL",             "10.0.0"),
    ("sklearn",         "1.3.0"),
    ("plotly",          "5.15.0"),
    ("matplotlib",      "3.7.0"),
    ("seaborn",         "0.12.0"),
    ("tqdm",            "4.65.0"),
    ("cv2",             "4.8.0"),
    ("scipy",           "1.11.0"),
]

print(f"Python: {sys.version}")
print()

ok = True
for pkg, min_ver in REQUIRED:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✓ {pkg:<20} {ver}")
    except ImportError:
        print(f"  ✗ {pkg:<20} NOT INSTALLED  (required >= {min_ver})")
        ok = False

print()
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu (training will be slower)'}")

if not ok:
    print("\n⚠  Some packages are missing. Run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✓ Environment OK")
