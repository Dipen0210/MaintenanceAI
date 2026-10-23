# Public repository boundaries

Publish application source, tests, synthetic fixtures, dependency manifests, sanitized reference benchmarks and user-facing documentation.

Keep credentials, environment files, gateway configuration downloads, private keys, customer records, uploaded manuals/photos, databases, model weights and raw datasets out of Git. Put new business plans, roadmaps and private notes in `private/` or `docs/private/`; these folders are ignored. Internal project planning files and generated audit/evaluation/UI records are also excluded by the root `.gitignore`.

`.gitignore` does not protect a file already tracked, prevent `git add -f`, or remove information from past commits. Before publishing changes, inspect `git diff --cached --name-status`, review the staged diff and run a secret scan. Do not paste real access keys into examples or fixtures. Documented demo credentials are for the local synthetic facility only.

An ignored file is not backed up by Git. Maintain a separate private, access-controlled backup of planning documents and local data.

With Gitleaks installed, run `gitleaks git --staged --redact` before committing and `gitleaks git --log-opts="--all --full-history" --redact` to inspect history. The repository configuration retains the default detectors and exempts only five verified tokenizer checksum values at their exact manifest path. Do not broadly exclude credential-bearing files from scanning.

The existing [MIT license](../LICENSE) applies to the published project. Private planning files are excluded from this publication; public source should not be treated as confidential business information. License changes require a separate explicit decision.
