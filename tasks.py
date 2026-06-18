import sys

from invoke import task


@task
def lint(c):
    c.run("ruff check .")


@task
def test(c):
    c.run(f'"{sys.executable}" -m pytest -q')


@task
def train(c):
    c.run("python src/train.py")
