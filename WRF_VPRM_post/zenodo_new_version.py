"""
zenodo_new_version.py
======================================================================
Create a NEW VERSION of a Zenodo record, replace one file, and leave the
result as an UNPUBLISHED DRAFT for a human to review and publish.

Deliberately does NOT publish. Publishing on Zenodo is irreversible: the
version cannot be deleted afterwards, only superseded, and it mints a DOI
immediately. The final click stays with you.

Credentials
-----------
Reads the token from the ZENODO_TOKEN environment variable ONLY -- never a
command-line argument (which would leak into your shell history and the
process list). Create one at https://zenodo.org/account/settings/
applications/tokens/new/ with scopes `deposit:write` and `deposit:actions`:

    export ZENODO_TOKEN=...        # in your own shell
    python zenodo_new_version.py --record 20395680 \
        --replace WRFOUT.zip --with /scratch/c7071034/DATA/zenodo_v0.3/WRFOUT.zip \
        --version v0.3 --changelog /scratch/.../WRFOUT/CHANGELOG_v0.3.md

Existing metadata (title, authors, license, ...) is preserved; only
`version` is set and the changelog is prepended to `description`. The
record's other files carry over from the previous version untouched.

Use --dry-run first: it authenticates, shows the current files and what
would change, and makes no modifications.
======================================================================
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://zenodo.org/api"


def req(url, token, method="GET", data=None, ctype="application/json"):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method=method)
    r.add_header("Authorization", f"Bearer {token}")
    if body:
        r.add_header("Content-Type", ctype)
    with urllib.request.urlopen(r) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def put_file(bucket_url, name, path, token):
    """Stream a large file into the deposition bucket."""
    size = os.path.getsize(path)
    print(f"  uploading {name}  ({size/1e9:.1f} GB) -- this will take a while")
    with open(path, "rb") as fh:
        r = urllib.request.Request(f"{bucket_url}/{name}", data=fh, method="PUT")
        r.add_header("Authorization", f"Bearer {token}")
        r.add_header("Content-Type", "application/octet-stream")
        r.add_header("Content-Length", str(size))
        with urllib.request.urlopen(r) as resp:
            return json.loads(resp.read())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", required=True, help="existing record id, e.g. 20395680")
    p.add_argument("--replace", required=True, help="filename in the record to replace")
    p.add_argument("--with", dest="newfile", required=True, help="local file to upload")
    p.add_argument("--version", required=True, help="new version string, e.g. v0.3")
    p.add_argument("--changelog", help="markdown file prepended to the description")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN is not set. Export it in your own shell; this script "
              "never accepts a token as an argument.", file=sys.stderr)
        return 2
    if not os.path.isfile(args.newfile):
        print(f"ERROR: no such file: {args.newfile}", file=sys.stderr)
        return 2

    try:
        dep = req(f"{API}/deposit/depositions/{args.record}", token)
    except urllib.error.HTTPError as e:
        print(f"ERROR: cannot access deposition {args.record}: HTTP {e.code}. "
              "Check the token owns this record and has deposit:write.", file=sys.stderr)
        return 1

    print(f"record  : {dep.get('title')}")
    print(f"current : version={dep.get('metadata',{}).get('version')}  doi={dep.get('doi')}")
    print("files   :")
    for f in dep.get("files", []):
        mark = "  <-- to be replaced" if f["filename"] == args.replace else ""
        print(f"   {f['filesize']/1e6:10.1f} MB  {f['filename']}{mark}")
    if args.replace not in [f["filename"] for f in dep.get("files", [])]:
        print(f"ERROR: '{args.replace}' is not in the record.", file=sys.stderr)
        return 1

    new_size = os.path.getsize(args.newfile)
    print(f"\nwould upload: {args.newfile}  ({new_size/1e9:.1f} GB)  as {args.replace}")
    print(f"would set   : version = {args.version}")
    if args.dry_run:
        print("\n(dry run -- nothing changed)")
        return 0

    print("\ncreating new version ...")
    nv = req(f"{API}/deposit/depositions/{args.record}/actions/newversion", token,
             method="POST")
    draft = req(nv["links"]["latest_draft"], token)
    did = draft["id"]
    print(f"  draft id {did}")

    for f in draft.get("files", []):
        if f["filename"] == args.replace:
            req(f"{API}/deposit/depositions/{did}/files/{f['id']}", token, method="DELETE")
            print(f"  removed old {args.replace}")

    put_file(draft["links"]["bucket"], args.replace, args.newfile, token)
    print("  upload complete")

    meta = dict(draft["metadata"])
    meta["version"] = args.version
    if args.changelog and os.path.isfile(args.changelog):
        with open(args.changelog) as fh:
            note = fh.read()
        meta["description"] = ("<pre>" + note + "</pre><hr>" +
                               meta.get("description", ""))
    req(f"{API}/deposit/depositions/{did}", token, method="PUT", data={"metadata": meta})
    print("  metadata updated")

    print(f"\nDRAFT READY -- NOT PUBLISHED.")
    print(f"Review and publish by hand at: https://zenodo.org/uploads/{did}")
    print("Publishing is irreversible and mints a DOI, so that click is yours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
