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

Omit --replace and --with entirely for a metadata-only new version -- a
version bump that keeps every file as it is, for records that must stay in
step with a release even though their content did not change.

Use --dry-run first: it authenticates, shows the current files and what
would change, and makes no modifications.
======================================================================
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://zenodo.org/api"
UPLOAD_TRIES = 4          # a 29 GB PUT over a long-lived TLS connection does drop
GET_TRIES = 5             # zenodo.org returns 502/503/504 often enough to matter
RETRY_STATUS = {500, 502, 503, 504}


def req(url, token, method="GET", data=None, ctype="application/json"):
    body = json.dumps(data).encode() if data is not None else None
    tries = GET_TRIES if method == "GET" else 1

    for attempt in range(1, tries + 1):
        r = urllib.request.Request(url, data=body, method=method)
        r.add_header("Authorization", f"Bearer {token}")
        if body:
            r.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(r) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            # Only transient server-side codes are worth repeating; 4xx never is.
            if e.code not in RETRY_STATUS or attempt == tries:
                raise
            why = f"HTTP {e.code}"
        except OSError as e:
            if attempt == tries:
                raise
            why = str(e)
        wait = 15 * attempt
        print(f"  {why} -- retrying in {wait}s [attempt {attempt + 1}/{tries}]",
              file=sys.stderr)
        time.sleep(wait)


def put_file_curl(bucket_url, name, path, token):
    """Upload via curl, which survives long transfers far better than urllib.

    The token goes in on stdin as a curl config file, never in argv, so it
    cannot be read out of the process list.
    """
    config = (f'url = "{bucket_url}/{name}"\n'
              f'header = "Authorization: Bearer {token}"\n'
              f'header = "Content-Type: application/octet-stream"\n'
              f'upload-file = "{path}"\n'
              'fail = true\n'
              'silent = true\n'
              'show-error = true\n'
              'retry = 3\n'
              'retry-delay = 30\n'
              'speed-time = 300\n'      # give up a stalled connection after
              'speed-limit = 1024\n')   # 5 min under 1 KB/s, so a retry can fire
    proc = subprocess.run(["curl", "-K", "-"], input=config, text=True,
                          capture_output=True)
    if proc.returncode != 0:
        raise OSError(f"curl exit {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def put_file_urllib(bucket_url, name, path, token):
    """Fallback transfer when curl is unavailable."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        r = urllib.request.Request(f"{bucket_url}/{name}", data=fh, method="PUT")
        r.add_header("Authorization", f"Bearer {token}")
        r.add_header("Content-Type", "application/octet-stream")
        r.add_header("Content-Length", str(size))
        with urllib.request.urlopen(r) as resp:
            return json.loads(resp.read())


def put_file(bucket_url, name, path, token):
    """Stream a large file into the deposition bucket."""
    size = os.path.getsize(path)
    via = "curl" if shutil.which("curl") else "urllib"
    print(f"  uploading {name}  ({size/1e9:.1f} GB) via {via} -- this will take a while")
    if via == "curl":
        return put_file_curl(bucket_url, name, path, token)
    return put_file_urllib(bucket_url, name, path, token)


def find_open_draft(dep, token):
    """Locate an unpublished draft of this record's concept.

    `links.latest_draft` on a published record can point back at that record
    itself, so it cannot be trusted to find an interrupted run's draft. The
    deposition list carries `conceptrecid` and `submitted`, which can.
    """
    concept = str(dep.get("conceptrecid") or "")
    if not concept:
        return None
    for d in req(f"{API}/deposit/depositions?size=100", token):
        if (str(d.get("conceptrecid")) == concept
                and d.get("submitted") is False
                and str(d.get("id")) != str(dep.get("id"))):
            return d
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", required=True, help="existing record id, e.g. 20395680")
    p.add_argument("--replace", help="filename in the record to replace "
                                     "(omit together with --with for a metadata-only version)")
    p.add_argument("--with", dest="newfile", help="local file to upload")
    p.add_argument("--version", required=True, help="new version string, e.g. v0.3")
    p.add_argument("--changelog", help="markdown file prepended to the description")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN is not set. Export it in your own shell; this script "
              "never accepts a token as an argument.", file=sys.stderr)
        return 2
    if bool(args.replace) != bool(args.newfile):
        print("ERROR: --replace and --with go together. Give both to swap a file, "
              "or neither for a metadata-only new version.", file=sys.stderr)
        return 2
    replacing = bool(args.replace)
    if replacing and not os.path.isfile(args.newfile):
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
    if replacing and args.replace not in [f["filename"] for f in dep.get("files", [])]:
        print(f"ERROR: '{args.replace}' is not in the record.", file=sys.stderr)
        return 1

    if replacing:
        new_size = os.path.getsize(args.newfile)
        print(f"\nwould upload: {args.newfile}  ({new_size/1e9:.1f} GB)  as {args.replace}")
    else:
        print("\nwould upload: nothing -- files carry over unchanged")
    print(f"would set   : version = {args.version}")
    if dep.get("links", {}).get("latest_draft"):
        print("note        : an unpublished draft already exists -- it will be reused, "
              "not created anew")
    if args.dry_run:
        print("\n(dry run -- nothing changed)")
        return 0

    # `latest_draft` on a published record can point back at that same record's
    # editable form. Reuse it ONLY when it is a genuinely separate, unpublished
    # deposition -- otherwise we would be editing the published version in place.
    draft = find_open_draft(dep, token)
    if draft is not None:
        print(f"\nreusing the existing unpublished draft {draft['id']} ...")
    else:
        print("\ncreating new version ...")
        try:
            nv = req(f"{API}/deposit/depositions/{args.record}/actions/newversion", token,
                     method="POST")
        except urllib.error.HTTPError as e:
            if e.code == 400:
                print("ERROR: Zenodo refuses a new version, which means a draft exists "
                      "that this script could not find. Open https://zenodo.org/me/uploads "
                      "and publish or discard it, then re-run.", file=sys.stderr)
                return 1
            raise
        draft = req(nv["links"]["latest_draft"], token)
    did = draft["id"]
    print(f"  draft id {did}")

    # Hard stop: never mutate files on the published record or on anything submitted.
    if str(did) == str(args.record) or draft.get("submitted") is not False:
        print(f"ERROR: refusing to touch deposition {did} -- it is the published record, "
              "not an unpublished draft. Nothing was changed.", file=sys.stderr)
        return 1

    if replacing:
        for attempt in range(1, UPLOAD_TRIES + 1):
            # Re-read the draft each time: a failed PUT can leave a partial object,
            # and the bucket link is only valid for the current draft state.
            draft = req(f"{API}/deposit/depositions/{did}", token)
            for f in draft.get("files", []):
                if f["filename"] == args.replace:
                    req(f"{API}/deposit/depositions/{did}/files/{f['id']}", token,
                        method="DELETE")
                    print(f"  removed old {args.replace}")
            try:
                put_file(draft["links"]["bucket"], args.replace, args.newfile, token)
                print("  upload complete")
                break
            except OSError as e:
                # URLError and the ssl.SSL* errors all derive from OSError.
                if attempt == UPLOAD_TRIES:
                    print(f"ERROR: upload failed {UPLOAD_TRIES}x, last error: {e}",
                          file=sys.stderr)
                    print(f"Draft {did} is still open -- re-run the same command and it "
                          "will pick up where this left off.", file=sys.stderr)
                    return 1
                wait = 60 * attempt
                print(f"  upload failed ({e})")
                print(f"  retrying in {wait}s  [attempt {attempt + 1}/{UPLOAD_TRIES}]")
                time.sleep(wait)
    else:
        print("  no file changes -- carried over from the previous version")

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
