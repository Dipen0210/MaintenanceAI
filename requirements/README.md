# Python dependencies

From the repository root, install with the project environment:

```sh
venv/bin/python -m pip install -r requirements/scheduling.txt
```

| File | Scope |
| --- | --- |
| [../requirements.txt](../requirements.txt) | Base API and predictive ML dependencies |
| [dev.txt](dev.txt) | Base plus backend test tools |
| [copilot.txt](copilot.txt) | Development plus language/document tooling |
| [scheduling.txt](scheduling.txt) | Copilot plus scheduling solver; complete demo |
| [industrial.txt](industrial.txt) | Development plus optional OPC UA tooling |
| [lock.txt](lock.txt) | Exact packages from the validated local environment |

Install `requirements/industrial.txt` additionally for OPC UA verification or forwarding. Includes are relative to each requirements file. The lock records the original platform; other platforms need compatible wheels and independent validation. Frontend dependencies stay in `frontend/package.json` and its lockfile.
