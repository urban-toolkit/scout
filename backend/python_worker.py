# python_worker.py
import sys
import json
import io
import contextlib
import traceback

# Shared namespace across all exec calls -> keeps imports/graphs/etc. in memory
GLOBAL_NS = {"__name__": "__main__"}

def handle_request(req):
    code = req.get("code", "")

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            exec(code, GLOBAL_NS, GLOBAL_NS)
            ok = True
        except Exception:
            traceback.print_exc(file=stderr_buf)
            ok = False

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
