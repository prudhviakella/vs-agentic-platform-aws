"""
clinical_trials_loader.py
==========================
Fetches clinical trial data from the ClinicalTrials.gov v2 API (free, public)
and loads it into your Neo4j AuraDB graph database.

SETUP:
  1. Create a free Neo4j AuraDB at https://neo4j.com/cloud/aura-free/
  2. Copy your connection URI and password from the AuraDB console
  3. Set environment variables (recommended) OR edit the config block below
  4. Install dependencies and run:

     pip install neo4j requests
     python3 clinical_trials_loader.py

ENVIRONMENT VARIABLES (recommended — keeps credentials out of code):
  export NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
  export NEO4J_USER=neo4j
  export NEO4J_PASSWORD=your-password-here

WHAT IT LOADS:
  25 clinical trials across oncology, immunology, cardiology, neurology,
  infectious disease covering COVID vaccines, CAR-T, diabetes, MS, and more.

  Each trial is loaded as a connected subgraph:
    Trial ──TARGETS──────► Disease
          ──USES──────────► Drug
          ──SPONSORED_BY──► Sponsor
          ──MANAGED_BY────► CRO
          ──CONDUCTED_IN──► Country
          ──LOCATED_AT────► Site ──IN_COUNTRY──► Country
          ──MEASURES──────► Outcome
          ──INCLUDES──────► PatientPopulation
          ──ASSOCIATED_WITH► MeSHTerm
          ──BELONGS_TO────► TrialCategory
"""

import os
import logging
import requests
from typing import Optional
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neo4j connection
# Option 1: set environment variables (recommended)
# Option 2: replace the os.environ.get(...) defaults with your values
# ---------------------------------------------------------------------------
URI  = os.environ.get("NEO4J_URI",      "neo4j+s://REPLACE_ME.databases.neo4j.io")
USER = os.environ.get("NEO4J_USER",     "neo4j")
PASS = os.environ.get("NEO4J_PASSWORD", "REPLACE_ME")
AUTH = (USER, PASS)

# ---------------------------------------------------------------------------
# 25 clinical trials — fetched live from ClinicalTrials.gov (free public API)
# ---------------------------------------------------------------------------
CLINICAL_TRIALS = [
    {"nct_id": "NCT04368728", "name": "Remdesivir_COVID"},
    {"nct_id": "NCT04470427", "name": "Pfizer_Vaccine"},
    {"nct_id": "NCT03235752", "name": "Ulcerative_Colitis"},
    {"nct_id": "NCT03961204", "name": "Classic_MS"},
    {"nct_id": "NCT03164772", "name": "Heart_Failure"},
    {"nct_id": "NCT04032704", "name": "CAR-T_Cell"},
    {"nct_id": "NCT03753074", "name": "Hepatitis_B_TAF"},
    {"nct_id": "NCT02014597", "name": "Scleroderma_Study"},
    {"nct_id": "NCT03181503", "name": "Lupus_Nephritis"},
    {"nct_id": "NCT03434379", "name": "Breast_Cancer"},
    {"nct_id": "NCT04652245", "name": "Janssen_COVID_Vax"},
    {"nct_id": "NCT04280705", "name": "Hydroxychloroquine_COVID"},
    {"nct_id": "NCT04614948", "name": "Moderna_Vaccine"},
    {"nct_id": "NCT03548935", "name": "Alzheimers_Trial"},
    {"nct_id": "NCT03155620", "name": "Parkinsons_Study"},
    {"nct_id": "NCT02968303", "name": "HIV_Treatment"},
    {"nct_id": "NCT03518606", "name": "Melanoma_Immuno"},
    {"nct_id": "NCT02863419", "name": "Lung_Cancer_NSCLC"},
    {"nct_id": "NCT02951156", "name": "Prostate_Cancer"},
    {"nct_id": "NCT03374254", "name": "Colon_Cancer"},
    {"nct_id": "NCT02788279", "name": "Leukemia_CAR_T"},
    {"nct_id": "NCT03544736", "name": "Multiple_Myeloma"},
    {"nct_id": "NCT03423992", "name": "Rheumatoid_Arthritis"},
    {"nct_id": "NCT02579382", "name": "Crohns_Disease"},
    {"nct_id": "NCT03662659", "name": "Type2_Diabetes"},
]

CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"


# ===========================================================================
# SECTION 1 — API FETCH
# ===========================================================================

def fetch_trial(nct_id: str) -> Optional[dict]:
    """Fetch raw study JSON for one NCT ID from ClinicalTrials.gov v2 API."""
    url     = f"{CT_API_BASE}/{nct_id}"
    headers = {"User-Agent": "clinical-trials-neo4j-loader/1.0 (research use)"}

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning(f"Attempt {attempt}/3 failed for {nct_id}: {exc}")

    log.error(f"All retries exhausted for {nct_id}")
    return None


def dig(data: dict, *keys, default=None):
    """Safely traverse a nested dict/list by a sequence of keys/indices."""
    for key in keys:
        if data is None:
            return default
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return default
    return data if data is not None else default


# ===========================================================================
# SECTION 2 — API RESPONSE → FLAT GRAPH RECORD
# ===========================================================================

def parse_trial(raw: dict) -> dict:
    """Transform the raw ClinicalTrials.gov v2 JSON into a flat graph record."""
    ps   = dig(raw, "protocolSection", default={})
    ident        = dig(ps, "identificationModule",        default={})
    status       = dig(ps, "statusModule",                default={})
    design       = dig(ps, "designModule",                default={})
    arms         = dig(ps, "armsInterventionsModule",     default={})
    spons        = dig(ps, "sponsorCollaboratorsModule",  default={})
    elig         = dig(ps, "eligibilityModule",           default={})
    locs         = dig(ps, "contactsLocationsModule",     default={})
    outcomes_mod = dig(ps, "outcomesModule",              default={})
    conds        = dig(ps, "conditionsModule",            default={})

    # derivedSection is a sibling of protocolSection at top level
    ds = dig(raw, "derivedSection", default={})
    mesh = (
        dig(ds, "conditionBrowseModule",    "meshes", default=[]) +
        dig(ds, "interventionBrowseModule", "meshes", default=[])
    )

    trial = {
        "nctId":                dig(ident,  "nctId"),
        "briefTitle":           dig(ident,  "briefTitle"),
        "officialTitle":        dig(ident,  "officialTitle"),
        "acronym":              dig(ident,  "acronym"),
        "overallStatus":        dig(status, "overallStatus"),
        "startDate":            dig(status, "startDateStruct",           "date"),
        "primaryCompletionDate":dig(status, "primaryCompletionDateStruct","date"),
        "completionDate":       dig(status, "completionDateStruct",      "date"),
        "studyFirstSubmitDate": dig(status, "studyFirstSubmitDate"),
        "lastUpdateSubmitDate": dig(status, "lastUpdateSubmitDate"),
        "statusVerifiedDate":   dig(status, "statusVerifiedDate"),
        "phase":                ", ".join(dig(design, "phases", default=[])),
        "enrollmentCount":      dig(design, "enrollmentInfo", "count"),
        "enrollmentType":       dig(design, "enrollmentInfo", "type"),
    }

    conditions    = dig(conds, "conditions", default=[])
    interventions = [
        {
            "name":       dig(iv, "interventionName"),
            "type":       dig(iv, "interventionType"),
            "otherNames": dig(iv, "otherNames", default=[]),
        }
        for iv in dig(arms, "interventions", default=[])
    ]
    lead_sponsor  = dig(spons, "leadSponsor", "name")
    collaborators = [
        dig(c, "name") for c in dig(spons, "collaborators", default=[])
        if dig(c, "name")
    ]
    locations = [
        {
            "facility": dig(loc, "facility"),
            "city":     dig(loc, "city"),
            "country":  dig(loc, "country"),
            "zip":      dig(loc, "zip"),
            "lat":      dig(loc, "geoPoint", "lat"),
            "lon":      dig(loc, "geoPoint", "lon"),
        }
        for loc in dig(locs, "locations", default=[])
    ]
    primary_outcomes = [
        {
            "measure":     dig(o, "measure"),
            "description": dig(o, "description", default=""),
            "timeFrame":   dig(o, "timeFrame",   default=""),
            "type":        "primary",
        }
        for o in dig(outcomes_mod, "primaryOutcomes", default=[])
    ]
    secondary_outcomes = [
        {
            "measure":   dig(o, "measure"),
            "timeFrame": dig(o, "timeFrame", default=""),
            "type":      "secondary",
        }
        for o in dig(outcomes_mod, "secondaryOutcomes", default=[])
    ]
    patient_population = {
        "eligibilityCriteria": dig(elig, "eligibilityCriteria"),
        "gender":              dig(elig, "sex"),
        "minimumAge":          dig(elig, "minimumAge"),
        "maximumAge":          dig(elig, "maximumAge"),
        "stdAges":             dig(elig, "stdAges", default=[]),
        "healthyVolunteers":   str(dig(elig, "healthyVolunteers", default="")),
    }
    mesh_terms = [dig(m, "term") for m in mesh if dig(m, "term")]

    return {
        "trial":              trial,
        "conditions":         conditions,
        "interventions":      interventions,
        "lead_sponsor":       lead_sponsor,
        "collaborators":      collaborators,
        "locations":          locations,
        "primary_outcomes":   primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "patient_population": patient_population,
        "mesh_terms":         mesh_terms,
    }


# ===========================================================================
# SECTION 3 — NEO4J SCHEMA SETUP
# ===========================================================================

def create_constraints_and_indexes(driver):
    """Create uniqueness constraints and lookup indexes before any data loads."""
    ddl = [
        "CREATE CONSTRAINT trial_nct_id     IF NOT EXISTS FOR (t:Trial)          REQUIRE t.nctId  IS UNIQUE",
        "CREATE CONSTRAINT category_name    IF NOT EXISTS FOR (tc:TrialCategory)  REQUIRE tc.name  IS UNIQUE",
        "CREATE CONSTRAINT disease_name     IF NOT EXISTS FOR (d:Disease)         REQUIRE d.name   IS UNIQUE",
        "CREATE CONSTRAINT drug_name        IF NOT EXISTS FOR (dr:Drug)           REQUIRE dr.name  IS UNIQUE",
        "CREATE CONSTRAINT sponsor_name     IF NOT EXISTS FOR (s:Sponsor)         REQUIRE s.name   IS UNIQUE",
        "CREATE CONSTRAINT cro_name         IF NOT EXISTS FOR (cro:CRO)           REQUIRE cro.name IS UNIQUE",
        "CREATE CONSTRAINT country_name     IF NOT EXISTS FOR (co:Country)        REQUIRE co.name  IS UNIQUE",
        "CREATE INDEX trial_status  IF NOT EXISTS FOR (t:Trial)    ON (t.overallStatus)",
        "CREATE INDEX trial_phase   IF NOT EXISTS FOR (t:Trial)    ON (t.phase)",
        "CREATE INDEX site_city     IF NOT EXISTS FOR (si:Site)    ON (si.city)",
        "CREATE INDEX mesh_term_idx IF NOT EXISTS FOR (m:MeSHTerm) ON (m.term)",
    ]
    with driver.session() as session:
        for stmt in ddl:
            try:
                session.run(stmt)
                log.info(f"DDL OK: {stmt[:70]}…")
            except Exception as exc:
                log.warning(f"DDL skipped (already exists): {exc}")


# ===========================================================================
# SECTION 4 — NODE / RELATIONSHIP WRITERS
# ===========================================================================

def load_trial_node(session, trial: dict):
    session.run("""
        MERGE (t:Trial {nctId: $nctId})
        SET t.briefTitle            = $briefTitle,
            t.officialTitle         = $officialTitle,
            t.acronym               = $acronym,
            t.overallStatus         = $overallStatus,
            t.startDate             = $startDate,
            t.phase                 = $phase,
            t.statusVerifiedDate    = $statusVerifiedDate,
            t.primaryCompletionDate = $primaryCompletionDate,
            t.completionDate        = $completionDate,
            t.studyFirstSubmitDate  = $studyFirstSubmitDate,
            t.lastUpdateSubmitDate  = $lastUpdateSubmitDate,
            t.enrollmentCount       = $enrollmentCount,
            t.enrollmentType        = $enrollmentType
    """, **trial)


def load_category(session, nct_id: str, conditions: list):
    category = conditions[0] if conditions else "Unknown"
    session.run("""
        MERGE (tc:TrialCategory {name: $category})
        WITH tc
        MATCH (t:Trial {nctId: $nctId})
        MERGE (t)-[:BELONGS_TO]->(tc)
    """, category=category, nctId=nct_id)


def load_diseases(session, nct_id: str, conditions: list):
    for condition in conditions:
        if condition and condition.strip():
            session.run("""
                MERGE (d:Disease {name: $name})
                WITH d
                MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:TARGETS]->(d)
            """, name=condition.strip(), nctId=nct_id)


def load_drugs(session, nct_id: str, interventions: list):
    for iv in interventions:
        name = (iv.get("name") or "").strip()
        if name:
            session.run("""
                MERGE (dr:Drug {name: $name})
                SET dr.type       = $type,
                    dr.otherNames = $otherNames
                WITH dr
                MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:USES]->(dr)
            """, name=name,
                 type=iv.get("type", ""),
                 otherNames=iv.get("otherNames", []),
                 nctId=nct_id)


def load_sponsor(session, nct_id: str, lead_sponsor: str):
    if lead_sponsor and lead_sponsor.strip():
        session.run("""
            MERGE (s:Sponsor {name: $name})
            WITH s
            MATCH (t:Trial {nctId: $nctId})
            MERGE (t)-[:SPONSORED_BY]->(s)
        """, name=lead_sponsor.strip(), nctId=nct_id)


def load_cros(session, nct_id: str, collaborators: list):
    for name in collaborators:
        if name and name.strip():
            session.run("""
                MERGE (cro:CRO {name: $name})
                WITH cro
                MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:MANAGED_BY]->(cro)
            """, name=name.strip(), nctId=nct_id)


def load_locations(session, nct_id: str, locations: list):
    for loc in locations:
        country  = (loc.get("country")  or "").strip()
        facility = (loc.get("facility") or "").strip()
        if country:
            session.run("""
                MERGE (co:Country {name: $country})
                WITH co
                MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:CONDUCTED_IN]->(co)
            """, country=country, nctId=nct_id)
            if facility:
                session.run("""
                    MERGE (si:Site {facility: $facility})
                    SET si.city = $city,
                        si.zip  = $zip,
                        si.lat  = $lat,
                        si.lon  = $lon
                    WITH si
                    MATCH (t:Trial {nctId: $nctId})
                    MATCH (co:Country {name: $country})
                    MERGE (t)-[:LOCATED_AT]->(si)
                    MERGE (si)-[:IN_COUNTRY]->(co)
                """, facility=facility,
                     city=loc.get("city"),
                     zip=loc.get("zip"),
                     lat=loc.get("lat"),
                     lon=loc.get("lon"),
                     nctId=nct_id,
                     country=country)


def load_outcomes(session, nct_id: str, primary: list, secondary: list):
    for outcome in primary + secondary:
        measure = (outcome.get("measure") or "").strip()
        if measure:
            session.run("""
                CREATE (o:Outcome {
                    measure:     $measure,
                    description: $description,
                    timeFrame:   $timeFrame,
                    type:        $type
                })
                WITH o
                MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:MEASURES]->(o)
            """, measure=measure,
                 description=outcome.get("description", ""),
                 timeFrame=outcome.get("timeFrame", ""),
                 type=outcome.get("type", ""),
                 nctId=nct_id)


def load_patient_population(session, nct_id: str, pp: dict):
    session.run("""
        CREATE (pp:PatientPopulation {
            eligibilityCriteria: $eligibilityCriteria,
            gender:              $gender,
            minimumAge:          $minimumAge,
            maximumAge:          $maximumAge,
            stdAges:             $stdAges,
            healthyVolunteers:   $healthyVolunteers
        })
        WITH pp
        MATCH (t:Trial {nctId: $nctId})
        MERGE (t)-[:INCLUDES]->(pp)
    """, eligibilityCriteria=pp.get("eligibilityCriteria"),
         gender=pp.get("gender"),
         minimumAge=pp.get("minimumAge"),
         maximumAge=pp.get("maximumAge"),
         stdAges=pp.get("stdAges", []),
         healthyVolunteers=pp.get("healthyVolunteers"),
         nctId=nct_id)


def load_mesh_terms(session, nct_id: str, mesh_terms: list):
    for term in mesh_terms:
        if term and term.strip():
            session.run("""
                MERGE (m:MeSHTerm {term: $term})
                WITH m
                MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:ASSOCIATED_WITH]->(m)
            """, term=term.strip(), nctId=nct_id)


# ===========================================================================
# SECTION 5 — ORCHESTRATION
# ===========================================================================

def load_record(driver, record: dict):
    """Write all nodes and relationships for one parsed trial."""
    nct_id = record["trial"].get("nctId")
    if not nct_id:
        log.warning("Skipping record with no nctId")
        return

    with driver.session() as session:
        try:
            load_trial_node(session,         record["trial"])
            load_category(session,           nct_id, record["conditions"])
            load_diseases(session,           nct_id, record["conditions"])
            load_drugs(session,              nct_id, record["interventions"])
            load_sponsor(session,            nct_id, record["lead_sponsor"])
            load_cros(session,               nct_id, record["collaborators"])
            load_locations(session,          nct_id, record["locations"])
            load_outcomes(session,           nct_id,
                          record["primary_outcomes"],
                          record["secondary_outcomes"])
            load_patient_population(session, nct_id, record["patient_population"])
            load_mesh_terms(session,         nct_id, record["mesh_terms"])

            log.info(
                f"  {nct_id} | "
                f"conditions={len(record['conditions'])} "
                f"drugs={len(record['interventions'])} "
                f"locations={len(record['locations'])} "
                f"mesh={len(record['mesh_terms'])} "
                f"outcomes={len(record['primary_outcomes'])+len(record['secondary_outcomes'])}"
            )
        except Exception as exc:
            log.error(f"Failed to load {nct_id}: {exc}")


def print_stats(driver):
    """Print node counts for all main labels."""
    labels = ["Trial", "Disease", "Drug", "Sponsor", "CRO",
              "Country", "Site", "Outcome", "MeSHTerm", "TrialCategory",
              "PatientPopulation"]
    with driver.session() as session:
        print("\n── Graph statistics ──────────────────────")
        for label in labels:
            count = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS c"
            ).single()["c"]
            print(f"  {label:<22} {count:>6}")
        print("──────────────────────────────────────────\n")


# ===========================================================================
# SECTION 6 — ENTRY POINT
# ===========================================================================

def main():
    # Validate credentials before doing any work
    if "REPLACE_ME" in URI or "REPLACE_ME" in PASS:
        print("\n❌  Set your Neo4j credentials first:")
        print("    export NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io")
        print("    export NEO4J_USER=neo4j")
        print("    export NEO4J_PASSWORD=your-password")
        print("\n    Or edit the URI / PASS variables at the top of this file.\n")
        return

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        log.info("Neo4j connection verified ✅")

        create_constraints_and_indexes(driver)

        for i, entry in enumerate(CLINICAL_TRIALS, 1):
            nct_id = entry["nct_id"]
            log.info(f"[{i}/{len(CLINICAL_TRIALS)}] Fetching {nct_id}  ({entry['name']})")

            raw = fetch_trial(nct_id)
            if raw is None:
                log.warning(f"No data returned for {nct_id} — skipping")
                continue

            record = parse_trial(raw)
            load_record(driver, record)

        print_stats(driver)
        log.info("All trials loaded ✅")


if __name__ == "__main__":
    main()
