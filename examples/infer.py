"""Run a JSON request using the installed LightDecision package."""
import sys
from lightdecision.cli import main

if __name__ == "__main__":
    main(["infer", *sys.argv[1:]])
