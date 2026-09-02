"""`python -m manifoldbt` -- the CLI, reachable without the PATH.

The `manifoldbt` console script lands in the interpreter's Scripts/ directory,
which on Windows is frequently not on PATH: pip itself warns about it at
install time, and a user typing `manifoldbt activate <code>` then gets
"'manifoldbt' is not recognized" and reads it as the product being broken.
`python -m manifoldbt` works wherever `python` does, which is the one thing
a user who just ran `pip install` is guaranteed to have.
"""
from manifoldbt.cli import main

if __name__ == "__main__":
    main()
