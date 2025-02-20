import os 
import shutil


def save_code(save_dir):
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shutil.copytree(
        project_dir,
        save_dir + "/code",
        ignore=shutil.ignore_patterns(
            "output*",
            "log*",
            "result*",
            "outputs*",
            "data*",
            "*.out",
            "models*",
            "checkpoint*",
            ".git",
            "*.pyc",
            ".idea",
            ".DS_Store",
        ),
    )

