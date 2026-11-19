"""Fetch pinned runtime textures; optionally retrieve reference PDFs locally.

No knife download is provided: that asset's redistribution terms are unresolved.
Nothing fetched here is executed. Downloads are size-bounded, SHA-256 verified,
and installed atomically beneath the workspace assets directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT=Path(__file__).resolve().parents[1]
HOSTS={'dl.polyhaven.org','cdnmdm.laticrete.com','metroceramics.com','www.metroceramics.com'}
MAX_BYTES=64*1024*1024

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def destination(root,relative):
    root=Path(root).resolve();base=root/'assets';candidate=(root/relative).resolve()
    if not base.resolve().is_relative_to(root) or not candidate.is_relative_to(base.resolve()) or candidate==base.resolve():
        raise ValueError('Asset destination must remain beneath workspace/assets')
    return candidate

def check_url(url):
    p=urlsplit(url)
    if p.scheme!='https' or p.hostname not in HOSTS or p.username or p.password or p.port not in (None,443):
        raise ValueError('Asset URL must use an approved HTTPS host')

class VerifiedRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_url(newurl) # reject before a redirected connection is opened
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def urlopen(request,timeout=60):
    return build_opener(VerifiedRedirect()).open(request,timeout=timeout)

def fetch(record,root):
    path=destination(root,record['path'])
    if path.exists():
        if digest(path)!=record['sha256']:raise ValueError(f'Existing asset differs from manifest; preserved: {path}')
        return False
    check_url(record['url']);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with urlopen(Request(record['url'],headers={'User-Agent':'Chora asset bootstrap'}),timeout=60) as response:
            check_url(response.url)
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as output:
                temporary=Path(output.name);h=hashlib.sha256();total=0
                while chunk:=response.read(1024*1024):
                    total+=len(chunk)
                    if total>MAX_BYTES:raise ValueError('Asset exceeds download size limit')
                    h.update(chunk);output.write(chunk)
        if h.hexdigest()!=record['sha256']:raise ValueError(f'Checksum mismatch: {record["path"]}')
        temporary.replace(path);temporary=None
        return True
    finally:
        if temporary:temporary.unlink(missing_ok=True)

def convert(record,root):
    from PIL import Image
    path=destination(root,record['path']);source=destination(root,record['source'])
    if path.exists():
        if digest(path)!=record['sha256']:raise ValueError(f'Existing derived asset differs; preserved: {path}')
        return False
    temporary=None;path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as output:temporary=Path(output.name)
        with Image.open(source) as im:im.save(temporary,format='PNG')
        if digest(temporary)!=record['sha256']:raise ValueError('PNG checksum mismatch; use the pinned Pillow version in requirements-lock.txt')
        temporary.replace(path);temporary=None
        return True
    finally:
        if temporary:temporary.unlink(missing_ok=True)

def run(root=ROOT,documents=False,check=False,manifest=None):
    manifest=manifest or json.loads((ROOT/'configs/assets.json').read_text())
    for r in manifest['downloads']:
        if r['group']=='documents' and not documents:continue
        if check:
            p=destination(root,r['path'])
            if not p.is_file() or digest(p)!=r['sha256']:raise ValueError(f'Missing or changed asset: {r["path"]}')
        else:fetch(r,root)
    for r in manifest['derived']:
        if check:
            p=destination(root,r['path'])
            if not p.is_file() or digest(p)!=r['sha256']:raise ValueError(f'Missing or changed asset: {r["path"]}')
        else:convert(r,root)
    print('Runtime texture assets verified'+('; reference documents verified' if documents else ''))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--documents',action='store_true');p.add_argument('--check',action='store_true');args=p.parse_args()
    run(documents=args.documents,check=args.check)
