# sealed/: the registry of sealed data files

A sealed file (a prediction, or the inputs of an experiment) is written
down and registered before the evidence that will judge it is looked at.
The seal records its sha256 and the date, so that it can be shown later
that the file did not change.

`registry.jsonl` is append-only: one JSON line per sealed file, with

* `file`: its path, relative to the repository root;
* `sha256`: the hash of its bytes;
* `date`: the UTC time of the seal;
* `note`: what the file holds or claims (in Spanish in the entries so far).

The nine entries so far (18-20 August 2026) seal the data stores of the
development phase in `data_v4/store/`: the per-pixel water-index series of
Villaviciosa and of the Westerschelde, EOT20 tide series at the mouth of
Villaviciosa, the distance along the channel, the base per-pixel fit used
as the template of the simulator, and a per-pixel ponding index. Those
stores are not in the repository
([`../data_v4/README.md`](../data_v4/README.md)); the registry is the public
record of their hashes.

The mechanism is `pyintertidal.seal`: `create_seal` refuses to seal a path
twice, `verify_seal` recomputes a hash and fails loudly on a mismatch, and
`verify_registry` re-checks every entry (no test calls it yet).
