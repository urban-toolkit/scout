# python_worker.py
import sys
import json
import io
import os
import contextlib
import traceback

# Shared namespace across all exec calls -> keeps imports/graphs/etc. in memory
GLOBAL_NS = {"__name__": "__main__"}

def handle_request(req):
    code = req.get("code", "")
    # Optional per-request working directory (e.g. a dataflow's compute
    # scratch dir, see _compute_scratch_dir in server.py) so a compute
    # model's code can use bare relative paths like "computed/A.geojson" -
    # restored in `finally` so it never leaks into the next request, even
    # though GLOBAL_NS itself is intentionally shared/persistent.
    cwd = req.get("cwd")
    orig_cwd = os.getcwd()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            if cwd:
                os.chdir(cwd)
            exec(code, GLOBAL_NS, GLOBAL_NS)
            ok = True
        except Exception:
            traceback.print_exc(file=stderr_buf)
            ok = False
        finally:
            if cwd:
                os.chdir(orig_cwd)

    return {
        "ok": ok,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
    }

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "__quit__":
            break

        try:
            req = json.loads(line)
        except Exception:
            # send parse error
            resp = {
                "ok": False,
                "stdout": "",
                "stderr": "Failed to parse JSON request",
            }
        else:
            resp = handle_request(req)

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
