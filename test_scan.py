"""Self-check for the test-folder scan logic. Run: python test_scan.py
(Forge modules are stubbed out — this only exercises the pure filesystem code.)
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

for _m in ("gradio", "PIL", "modules", "modules.infotext_utils", "modules_forge"):
    sys.modules[_m] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import batch_hires_fix as bhf


def touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


with tempfile.TemporaryDirectory() as root:
    tests = os.path.join(root, "Commission 1 - A", "Tests")
    touch(tests, "1r1.png"); touch(tests, "1r1-adetailer.png")     # pending
    touch(tests, "2r1.png"); touch(tests, "2r1-adetailer.png")
    touch(tests, "2r1-adetailer-hires.png")                        # hires-fixed -> done
    touch(tests, "2r1-adetailer-base.png")                         # lanczos twin -> not an input
    touch(tests, "3r1.png"); touch(tests, "3r1-adetailer.png")
    touch(tests, "3r1-adetailer-edited.png")                       # edited past -> done
    touch(tests, "4r1.png")                                        # base only -> not ready yet
    touch(tests, "10r1.png"); touch(tests, "10r1-adetailer.jpg")   # pending; after 1r1, natural order
    touch(tests, "1r1-adetailer-1.png")                            # collision copy -> ignored
    touch(tests, "notes.txt")                                      # not an image -> ignored
    pending = [os.path.basename(p) for p in bhf._pending_adetailer(tests)]
    assert pending == ["1r1-adetailer.png", "10r1-adetailer.jpg"], pending

    touch(root, "Commission 2 - B", "Tests", "1r1-adetailer.png")  # 1 pending
    done = os.path.join(root, "Commission 3 - C", "Tests")
    touch(done, "1r1-adetailer.png"); touch(done, "1r1-adetailer-hires.png")  # done -> hidden
    touch(root, "Commission 4 - D", "readme.txt")                  # no Tests dir -> hidden

    bhf.shared.opts.batch_hires_fix_scan_roots = root + ";" + os.path.join(root, "missing")
    choices = bhf._scan_test_folders()
    assert [c[1] for c in choices] == [tests, os.path.join(root, "Commission 2 - B", "Tests")], choices
    assert "(2 to do)" in choices[0][0] and "(1 to do)" in choices[1][0], choices

print("ok")
