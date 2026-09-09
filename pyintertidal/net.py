"""
net.py — One place for the network quirks every remote call hits
================================================================

Three parts of this library talk to the internet: the openEO backend
(:mod:`pyintertidal.scenes`), the Overpass API for satellite overpass times
(:mod:`pyintertidal.overpass`), and the basemap tiles behind
:func:`pyintertidal.viz.plot_aoi`. They all trip over the same thing on a
managed machine, so the fix lives here once instead of three times.
"""

from __future__ import annotations

import os
import ssl

#: The real :class:`ssl.SSLContext`, captured at import. ``truststore``
#: substitutes the name with a subclass of itself, and once that has
#: happened the original is no longer reachable — so it has to be taken
#: here, before anything in this package has had a chance to inject.
_ORIGINAL_SSL_CONTEXT = ssl.SSLContext

_INJECTED = False
_BUNDLE = None


def use_system_certificates():
    """Validate TLS against the OPERATING SYSTEM's trust store.

    Python ships its own CA bundle, which on a corporate or university
    Windows machine does not contain the root certificate that the network's
    TLS inspection presents. Every HTTPS call then fails with
    ``CERTIFICATE_VERIFY_FAILED`` even though the connection is fine and the
    machine's browser opens the same URL happily.

    ``truststore`` redirects validation to the OS store, which does have that
    root. Safe to call repeatedly — the injection only happens once — and a
    no-op where ``truststore`` is not installed, so nothing here is a hard
    dependency.

    Returns
    -------
    bool
        Whether system certificates are in use.
    """
    global _INJECTED
    if _INJECTED:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _INJECTED = True
    except ImportError:
        pass
    return _INJECTED


def patch_botocore_ssl():
    """Stop ``botocore`` recursing to death when truststore is installed.

    ``truststore`` works by making :class:`ssl.SSLContext` a subclass of
    itself. ``botocore`` builds its own context and then does
    ``context.options |= ...``; with the substituted class in place, that
    setter calls itself until Python runs out of stack. Anything speaking S3
    through botocore — which is how ``copernicusmarine`` fetches CMEMS data
    — dies there.

    The fix is narrow on purpose: restore the original class *inside*
    botocore's context factory only, so the S3 transport gets a plain
    context while every other request still validates against the OS store.

    Idempotent, and a no-op if botocore is not installed.
    """
    try:
        import botocore.httpsession as session
    except ImportError:
        return False
    if getattr(session.create_urllib3_context, "_pyintertidal", False):
        return True

    original = _ORIGINAL_SSL_CONTEXT
    inner = session.create_urllib3_context

    def create_urllib3_context(*args, **kwargs):
        current = ssl.SSLContext
        ssl.SSLContext = original
        try:
            return inner(*args, **kwargs)
        finally:
            ssl.SSLContext = current

    create_urllib3_context._pyintertidal = True
    session.create_urllib3_context = create_urllib3_context
    return True


def system_ca_bundle(path=None):
    """Export the OS trust store to a PEM file and point the env at it.

    The companion to :func:`use_system_certificates`, for libraries that do
    NOT tolerate having ``ssl.SSLContext`` patched underneath them.
    ``botocore`` — which is what ``copernicusmarine`` speaks S3 through —
    builds its own SSL context by subclassing, and with truststore in place
    that recurses until Python gives up. Handing it a CA *file* through the
    standard environment variables sidesteps the machinery entirely.

    Windows only in practice: elsewhere ``certifi`` already matches the
    system store and this is a no-op returning None.

    Parameters
    ----------
    path : str, optional
        Where to write the bundle; defaults to ``~/.pyintertidal/ca.pem``.

    Returns
    -------
    str | None
        The bundle path, or None where the platform needs no help.
    """
    global _BUNDLE
    if _BUNDLE:
        return _BUNDLE

    import ssl

    if not hasattr(ssl, "enum_certificates"):        # not Windows
        return None

    path = path or os.path.join(os.path.expanduser("~"), ".pyintertidal",
                                "ca.pem")
    if not os.path.exists(path):
        pem = []
        for store in ("ROOT", "CA"):
            try:
                for cert, encoding, _trust in ssl.enum_certificates(store):
                    if encoding == "x509_asn":
                        pem.append(ssl.DER_cert_to_PEM_cert(cert))
            except Exception:
                continue
        if not pem:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as fh:
            fh.write("".join(pem))

    # The three variables between them cover requests, botocore and anything
    # that reads OpenSSL's default.
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "AWS_CA_BUNDLE"):
        os.environ.setdefault(var, path)
    _BUNDLE = path
    return path
