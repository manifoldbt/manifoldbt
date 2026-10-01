# Security Policy

## Supported versions

Security fixes are issued for the latest published release of `manifoldbt` on
PyPI. Older versions are not patched; upgrade with `pip install -U manifoldbt`.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Use GitHub's private reporting: go to the **Security** tab of this repository
and click **Report a vulnerability**. That opens a private advisory visible only
to you and the maintainers.

If you cannot use that, email **contact@manifoldbt.com** with `SECURITY` in the
subject line.

Please include:

- the version of `manifoldbt` and of Python, and the operating system
- what an attacker can do with the issue
- a minimal reproduction, ideally a script that runs against public data

## What to expect

- Acknowledgement within 5 business days.
- An assessment, with a fix timeline or an explanation of why it is not a
  vulnerability, within 15 business days.
- Credit in the release notes when a report leads to a fix, unless you prefer
  otherwise.

Please give us a reasonable window to publish a fix before disclosing publicly.

## Scope

In scope: the published `manifoldbt` package, its data connectors, and its CLI.

Of particular interest: anything that lets untrusted input reach code execution.
The library parses files and network responses that a user did not write,
including CSV imports and connector payloads, and it deserializes stored data.

Out of scope: the website, third-party forks and redistributions, and findings
that require an attacker to already control the machine running the backtest.
