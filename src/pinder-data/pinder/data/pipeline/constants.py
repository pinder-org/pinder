PROCESSED_STATUS_CODES = [
    "complete",
    "no metadata",
    "assembly failed",
    "entities failed",
    "non-protein assembly",
    "too many chains",
]

UNIPROT_UNDEFINED: str = "UNDEFINED"

CONSIDER_LEAKED: str = "too_many_neighbors"

# track specific list of pinder IDs which should not be considered as valid test members,
# with a reason why they are excluded.
TEST_SYSTEM_BLACKLIST: dict[str, str] = {
    "2xuw__A1_Q72HW2--2xuw__A2_Q72HW2": "non-biological assembly with borderline prodigy-cryst probability",
    "1hmc__A1_P09603--1hmc__B1_P09603": "calpha-only dimer",
}
