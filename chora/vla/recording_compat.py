"""Recognize the audited source transition for archived standard-bowl recordings.

Adding the rail and shelf changes the shared source hashes without changing the
standard bowl's observations, actions or physics. All other provenance fields
must still match, and replay still checks every saved physics state.
"""

# (hash before the new scenes, hash after the new scenes). A subsequent source
# change requires a fresh audit. In particular, this is not a wildcard for
# configuration, assets, camera settings, environment, or runtime differences.
STANDARD_BOWL_SOURCE_TRANSITION = {
    "chora/sim.py": (
        "3aba91a78c527ee919ca57fd4cf17fffdd1c236f8e7b1b1fb363288d297421a7",
        "03d53b3a763b79e97df478d4771683fecdc3689fdd8b1d50c1a209f8efe299fb",
    ),
    "chora/policy.py": (
        "844291e1ba62033da70c8e55efd49cc2912aa29bbe858850da1819ef9420cbd0",
        "9515734252dc7a923cb46c45e53dfc5700b1030c5ce7712f95bbc904f7f8dcea",
    ),
    "chora/vla/record.py": (
        "db898979b7f6b9bfda733979d8d616eed9adc94d95f57b7284f3f72c20b66256",
        "19db6c18d35cf7c44811166d8de99e41f9ddcab3f8d2d8b6c4ffb32c61300753",
    ),
}


def match_standard_bowl_sources(current, archived):
    if all(archived.get(path) == before and current.get(path) == after
           for path, (before, after) in STANDARD_BOWL_SOURCE_TRANSITION.items()):
        current.update({path: before for path, (before, _) in STANDARD_BOWL_SOURCE_TRANSITION.items()})
