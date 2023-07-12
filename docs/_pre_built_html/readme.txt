This directory contains pre-built docs to be uploaded to readthedocs.
This removes the requirement that RTD can build VSPEC (particularly the examples)
correctly. So long as you can build them locally, we are good.

Warning: These directories are very large. Do not update them unless absolutely necessary.
It is recommended that for a new version of VSPEC we created a new dir with that version's
name and change the path in readthedocs.yaml to point to that dir.