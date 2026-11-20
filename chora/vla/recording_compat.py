"""Recognize the audited source transition for archived standard-bowl recordings.

Adding and refining the rail and shelf changes the shared source hashes without changing the
standard bowl's observations, actions or physics. All other provenance fields
must still match, and replay still checks every saved physics state.
"""

# Each column identifies an audited source revision. A subsequent source
# change requires a fresh audit. In particular, this is not a wildcard for
# configuration, assets, camera settings, environment, or runtime differences.
STANDARD_BOWL_SOURCE_PROFILES = {
    "chora/sim.py": (
        "3aba91a78c527ee919ca57fd4cf17fffdd1c236f8e7b1b1fb363288d297421a7",
        "03d53b3a763b79e97df478d4771683fecdc3689fdd8b1d50c1a209f8efe299fb",
        "75db76cd68166ec5815d6c9bc3dc2321bb790bcd4a1aabdb0dc7aa31af636bad",
    ),
    "chora/policy.py": (
        "844291e1ba62033da70c8e55efd49cc2912aa29bbe858850da1819ef9420cbd0",
        "9515734252dc7a923cb46c45e53dfc5700b1030c5ce7712f95bbc904f7f8dcea",
        "5ba6f572716b5362d6ebfb5faadba0e37e91eb00da44849a413384b24264f6c7",
    ),
    "chora/vla/record.py": (
        "db898979b7f6b9bfda733979d8d616eed9adc94d95f57b7284f3f72c20b66256",
        "19db6c18d35cf7c44811166d8de99e41f9ddcab3f8d2d8b6c4ffb32c61300753",
        "19db6c18d35cf7c44811166d8de99e41f9ddcab3f8d2d8b6c4ffb32c61300753",
    ),
}


def match_standard_bowl_sources(current, archived):
    profiles = STANDARD_BOWL_SOURCE_PROFILES
    if not all(current.get(path) == versions[-1] for path, versions in profiles.items()):
        return
    for revision in (0, 1):
        if all(archived.get(path) == versions[revision] for path, versions in profiles.items()):
            current.update({path: versions[revision] for path, versions in profiles.items()})
            return
