# Contributing

Thanks for reading the code closely enough to want to change it.

This repository is a **published mirror**. `manifoldbt` is built from a private
Rust codebase; what you see here is the Python package, the examples, the
documentation and the benchmark harnesses, republished at every release. That
shapes what a pull request can usefully touch, so please read the next section
before you start work.

## Where a pull request can land

**Maintained here. Pull requests welcome.**

| Path          | What it is                                                     |
| ------------- | -------------------------------------------------------------- |
| `benchmarks/` | Cross-engine benchmark and parity harnesses, and their reports |
| `tests/`      | Smoke tests that run against the published wheel               |

For the workflows under `.github/workflows/`, open an issue first. They run
with repository secrets, so changes to them are written by maintainers from a
described fix rather than merged from a branch.

**Regenerated at every release. A pull request here is overwritten without
warning.**

`python/manifoldbt/`, `python/tests/`, `examples/`, `docs/`, `README.md`,
`pyproject.toml`, `LICENSE`, `SECURITY.md`, `.github/workflows/ci.yml`,
`.github/ISSUE_TEMPLATE/`.

The release sync copies these from the private repository. The copy wins every
time: no merge conflict, no notification, your commit simply stops existing at
the next release. If you have found a bug in one of them, **open an issue
instead**. The fix has to be made upstream and an issue is how it gets there.
Describe what you would have changed and we will carry it across.

The engine itself takes no pull requests. Bug reports against it are welcome and
useful: give a version, an environment, and a reproduction that runs.

## Running the tests

The root `pyproject.toml` sets `testpaths = ["python/tests"]`, so a bare
`pytest` collects the mirrored package suite and nothing else. Anything living
beside the code it covers needs its path spelled out:

```bash
pytest benchmarks/<harness>/ -q
```

If you add tests outside `python/tests/`, say so in the pull request and give
the command that runs them. Tests that no command reaches will rot unnoticed.

## License

The project is published under Apache 2.0 with the Commons Clause.
**Contributions are accepted under Apache 2.0 alone, without the Commons
Clause.**

By opening a pull request you agree that your contribution is licensed to the
project under Apache 2.0, including the patent grant in its section 3, and that
it may be used commercially, including in paid editions of `manifoldbt`.

If you cannot grant that, say so in the pull request and we will settle it
before merging rather than after.

## AI-assisted contributions

Allowed, on one condition: **you have read the result, you understand it, and
you can explain it.**

You are accountable for what you submit, whatever wrote it. Verify the claims,
actually run the checks you say you ran, and confirm the change fits the code
around it. Do not send a first pass for a maintainer to finish, and do not let
an agent open, edit or comment on anything here without you deciding to.

Assisted work is held to exactly the same standard as any other, and it often
clears it comfortably: the same tooling that drafts a change makes it cheap to
test it properly and to go looking for the ways it is wrong.

## Before you open it

- One change per pull request.
- Say what is wrong, what the change does, and how you checked it. Measurements
  beat adjectives.
- If it is a judgement call rather than a defect, give the argument against it
  too. That gets a pull request read faster, not slower.

## Security

Never in a public issue or pull request. See [SECURITY.md](SECURITY.md).
