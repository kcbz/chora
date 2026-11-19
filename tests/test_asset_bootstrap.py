import hashlib
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fetch_assets as assets

def test_asset_destinations_cannot_escape_workspace(tmp_path):
    (tmp_path/'assets').mkdir()
    assert assets.destination(tmp_path,'assets/texture.png')==tmp_path/'assets/texture.png'
    for bad in ['../secret','/tmp/secret','assets/../config','assets']:
        with pytest.raises(ValueError):assets.destination(tmp_path,bad)
    (tmp_path/'assets/link').symlink_to(tmp_path.parent,target_is_directory=True)
    with pytest.raises(ValueError):assets.destination(tmp_path,'assets/link/secret')

@pytest.mark.parametrize('url',['http://dl.polyhaven.org/a','https://attacker.example/a','https://' + 'test-user' + ':' + 'placeholder' + '@dl.polyhaven.org/a','https://dl.polyhaven.org:444/a'])
def test_download_host_and_scheme_are_restricted(url):
    with pytest.raises(ValueError):assets.check_url(url)

def test_changed_local_asset_is_not_overwritten(tmp_path):
    path=tmp_path/'assets/local.jpg';path.parent.mkdir();path.write_bytes(b'customized')
    record=dict(path='assets/local.jpg',url='https://dl.polyhaven.org/test',sha256=hashlib.sha256(b'expected').hexdigest())
    with pytest.raises(ValueError,match='preserved'):assets.fetch(record,tmp_path)
    assert path.read_bytes()==b'customized'

def test_bad_download_never_becomes_a_live_asset(tmp_path,monkeypatch):
    from io import BytesIO
    class Response(BytesIO):url='https://dl.polyhaven.org/test'
    monkeypatch.setattr(assets,'urlopen',lambda *args,**kwargs:Response(b'incorrect'))
    record=dict(path='assets/test.jpg',url='https://dl.polyhaven.org/test',sha256=hashlib.sha256(b'expected').hexdigest())
    with pytest.raises(ValueError,match='Checksum'):assets.fetch(record,tmp_path)
    assert not list((tmp_path/'assets').iterdir())


def test_redirects_are_validated_before_opening_the_destination():
    from urllib.request import Request
    with pytest.raises(ValueError):
        assets.VerifiedRedirect().redirect_request(Request('https://dl.polyhaven.org/test'),None,302,'Found',{},'http://127.0.0.1/private')
