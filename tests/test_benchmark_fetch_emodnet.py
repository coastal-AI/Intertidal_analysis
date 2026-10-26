# -*- coding: utf-8 -*-
"""fetch emodnet: request building, the manifest, the cache and the channel
anchors on a local fake GeoTIFF; failures return a non-zero code. No network."""
import json
import os

import numpy as np
import pytest

from experiments.benchmark.fetch import emodnet

SITE = "escalda"
X0, Y0, PX, H, W = 560_000.0, 5_700_000.0, 10.0, 30, 50
CRS = "EPSG:32631"


def grid():
    return {"xs": X0 + PX / 2 + PX * np.arange(W), "ys": Y0 - PX / 2 - PX * np.arange(H),
            "crs": CRS}


def fake_tiff(bbox, west_value=-5.0, east_value=1.0, split=0.5):
    """A DTM over the requested box: ``west_value`` west of the box's
    ``split`` share of longitudes, ``east_value`` east of it; one NaN cell."""
    from rasterio.transform import from_origin

    lon0, lon1, lat0, lat1 = bbox
    step = 0.0005
    nx, ny = int(np.ceil((lon1 - lon0) / step)), int(np.ceil((lat1 - lat0) / step))
    a = np.full((ny, nx), east_value, np.float32)
    a[:, : int(split * nx)] = west_value
    a[0, 0] = np.nan
    return emodnet._bytes_of_geotiff(a, from_origin(lon0, lat1, step, step)), a, (lon0, lat1, step)


class Resp:
    def __init__(self, content, status=200, ctype="image/tiff"):
        self.content, self.status_code, self.headers = content, status, {"Content-Type": ctype}


class Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, list(params)))
        if self.exc is not None:
            raise self.exc
        return self.resp


def test_request_params_and_url():
    bbox = (3.1, 3.4, 51.3, 51.5)
    p = emodnet.request_params(bbox)
    d = {}
    for k, v in p:
        d.setdefault(k, []).append(v)
    assert d["coverageId"] == ["emodnet__mean"] and d["format"] == ["image/tiff"]
    assert d["version"] == ["2.0.1"] and d["request"] == ["GetCoverage"]
    assert d["subset"] == ["Lat(51.300000,51.500000)", "Long(3.100000,3.400000)"]
    url = emodnet.request_url(p)
    assert url.startswith("https://ows.emodnet-bathymetry.eu/wcs?") and "emodnet__mean" in url
    # the frame box is padded by 2 km
    b = emodnet.bbox_lonlat(grid())
    assert b[0] < b[1] and b[2] < b[3]
    assert (b[1] - b[0]) * 111.32 * np.cos(np.radians(b[2])) > (W * PX / 1000.0 + 3.9)
    assert emodnet.CONFIG["vertical_datum"] == "LAT" and "DTM 2024" in emodnet.CONFIG["product"]


def test_fetch_manifest_anchors_and_cache(tmp_path):
    from pyproj import Transformer

    root = str(tmp_path / "v2")
    bbox = emodnet.bbox_lonlat(grid())
    data, a, (lon0, lat1, step) = fake_tiff(bbox)
    sess = Session(Resp(data))
    assert emodnet.run([SITE], root=root, session=sess, grids={SITE: grid()}) == 0
    assert len(sess.calls) == 1 and sess.calls[0][0] == emodnet.WCS_URL
    d = emodnet.site_dir(SITE, root)
    man = emodnet.read_manifest(d)
    assert man["status"] == "ok" and man["coverage_id"] == "emodnet__mean"
    assert man["file"]["sha256"] == emodnet.sha256_file(os.path.join(d, "dtm.tif"))
    assert man["raster"]["crs"] == "EPSG:4326" and man["raster"]["min"] == -5.0
    assert man["params"] == [list(p) for p in emodnet.request_params(bbox)] and man["utc"]
    # channel anchors: the frame pixels whose nearest DTM cell is <= -2 m (LAT)
    anc = np.load(emodnet.anchors_path(SITE, root))
    g = grid()
    X, Y = np.meshgrid(g["xs"], g["ys"])
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(X, Y)
    col = np.floor((lon - lon0) / step).astype(int)
    row = np.floor((lat1 - lat) / step).astype(int)
    want = np.flatnonzero((a[row, col] <= -2.0).ravel())
    np.testing.assert_array_equal(anc, want)
    assert 0 < len(anc) < H * W and man["channel_anchors"]["n_px"] == len(anc)
    elev = emodnet.dtm_on_grid(emodnet.verified_dtm(d)[0], g)
    np.testing.assert_array_equal(elev, a[row, col])
    # a second run reuses the cached tile (sha256 matches): no request
    sess2 = Session(exc=AssertionError("must not be called"))
    assert emodnet.run([SITE], root=root, session=sess2, grids={SITE: grid()}) == 0
    assert sess2.calls == []
    # a changed file is fetched again
    with open(os.path.join(d, "dtm.tif"), "ab") as f:
        f.write(b"x")
    with pytest.raises(emodnet.EmodnetError):
        emodnet.verified_dtm(d)
    sess3 = Session(Resp(data))
    assert emodnet.run([SITE], root=root, session=sess3, grids={SITE: grid()}) == 0
    assert len(sess3.calls) == 1 and emodnet.verified_dtm(d)[1]["status"] == "ok"


@pytest.mark.parametrize("resp", [
    Resp(b"oops", status=500),
    Resp(b'<?xml version="1.0"?><ows:ExceptionReport>no such coverage</ows:ExceptionReport>',
         ctype="application/xml"),
    Resp(b"not a tiff at all"),
])
def test_refused_requests_return_nonzero(tmp_path, resp):
    root = str(tmp_path / "v2")
    assert emodnet.run([SITE], root=root, session=Session(resp), grids={SITE: grid()}) == 1
    man = emodnet.read_manifest(emodnet.site_dir(SITE, root))
    assert man["status"] == "not available" and man["reason"]
    assert not os.path.exists(os.path.join(emodnet.site_dir(SITE, root), "dtm.tif"))
    assert not os.path.exists(emodnet.anchors_path(SITE, root))


def test_network_failure_returns_nonzero(tmp_path):
    import requests

    root = str(tmp_path / "v2")
    sess = Session(exc=requests.ConnectionError("no network"))
    assert emodnet.run([SITE], root=root, session=sess, grids={SITE: grid()}) == 1
    man = emodnet.read_manifest(emodnet.site_dir(SITE, root))
    assert man["status"] == "not available" and "ConnectionError" in man["reason"]
    with pytest.raises(emodnet.EmodnetError):
        emodnet.verified_dtm(emodnet.site_dir(SITE, root))
    # no frame grid (no record, no stores): nothing is requested
    sess = Session(exc=AssertionError("must not be called"))
    assert emodnet.run(["ferrol"], root=root, cubes_root=str(tmp_path / "nocubes"),
                       session=sess) == 1
    assert sess.calls == []
    json.dumps(emodnet.CONFIG)
