"""
Dataset definitions for the Sodir REST API.

This module contains the mapping of dataset names to API layer/table IDs.
The API is built on ArcGIS FeatureServer and provides two main services:

1. DataService - Clean tabular data with consistent naming
2. FactMaps - Display-oriented layers with pre-filtered views

Note: Some API table names don't match FactPages downloads!
See comments for important naming differences.
"""

# =============================================================================
# DataService Layers (with geometry)
# =============================================================================
LAYERS = {
    # Administrative boundaries
    "block": 1001,
    "quadrant": 1002,
    "sub_area": 1003,
    "sbm_block": 1004,  # Seabed minerals
    "sbm_quadrant": 1005,
    "areastatus": 1100,

    # Structural geology
    "structural_elements": 2000,
    "domes": 2001,
    "faults_boundaries": 2002,
    "sediment_boundaries": 2004,

    # Licensing areas
    "licence": 3000,
    "licence_area_history": 3002,
    "licence_document_area": 3006,
    "licence_area_count": 3011,
    "apa_gross": 3102,
    "apa_open": 3103,
    "announced_blocks_history": 3104,
    "announced_history": 3106,
    "apa_gross_history": 3107,
    "afex_area": 3200,
    "afex_area_history": 3201,
    "business_arrangement_area": 3300,
    "business_arrangement_history": 3301,

    # Seismic surveys
    "seismic_acquisition": 4000,
    "seismic_acquisition_poly": 4008,
    "sbm_sample_point": 4501,
    "sbm_survey_area": 4502,
    "sbm_survey_line": 4503,
    "sbm_survey_sub_area": 4504,

    # Core entities
    "wellbore": 5000,
    "facility": 6000,
    "pipeline": 6100,
    "discovery": 7000,
    "discovery_map_reference": 7004,
    "discovery_history": 7005,
    "field": 7100,
    "play": 7800,

    # Seabed minerals
    "sbm_occurrence": 8001,
    "sbm_play_resource_estimate": 8002,
}


# =============================================================================
# DataService Tables (no geometry)
# =============================================================================
TABLES = {
    # Company information
    "company": 1200,

    # Stratigraphy
    "strat_litho": 2100,
    "strat_litho_wellbore": 2101,  # Formation tops with depths!
    "strat_litho_wellbore_core": 2102,
    "strat_chrono": 2200,

    # Licence details
    "licence_additional_area": 3001,
    "licence_transfer_history": 3003,
    "licence_document": 3005,
    "licence_licensee_history": 3007,
    "licence_operator_history": 3008,
    "licence_phase_history": 3009,
    "licence_task": 3010,
    "licensing_activity": 3100,

    # Business arrangement details
    "business_arrangement_operator": 3302,
    "business_arrangement_licensee_history": 3304,
    "business_arrangement_transfer_history": 3305,

    # Petroleum register
    "petreg_licence": 3400,
    "petreg_licence_licensee": 3401,
    "petreg_licence_message": 3402,
    "petreg_licence_operator": 3403,

    # Seismic details
    "seismic_acquisition_area": 4001,
    "seismic_acquisition_country": 4002,
    "seismic_acquisition_progress": 4003,
    "seismic_acquisition_survey_type": 4006,
    "seismic_acquisition_2d": 4010,
    "seismic_acquisition_3d": 4011,
    "seismic_acquisition_4d": 4012,

    # Wellbore details
    # IMPORTANT: API table names differ from FactPages!
    "wellbore_casing": 5001,
    "wellbore_core": 5002,  # CO2 samples (94 records), NOT core data!
    "wellbore_core_photo": 5003,  # Actual core sample data (8,500+ records)
    "wellbore_document": 5004,
    "wellbore_dst": 5005,
    "wellbore_exploration_all": 5006,
    "wellbore_formation_top": 5007,  # Document metadata, NOT formation tops!
    "wellbore_history": 5008,
    "wellbore_mud": 5009,
    "wellbore_oil_sample": 5010,
    "wellbore_litho_strat": 5011,
    "wellbore_log": 5013,
    "wellbore_chrono_strat": 5014,
    "wellbore_coordinates": 5015,
    "wellbore_development_all": 5050,

    # Facility details
    "facility_history": 6001,
    "tuf": 6200,
    "tuf_owner": 6201,
    "tuf_operator": 6202,

    # Discovery details
    "discovery_area": 7001,
    "discovery_operator": 7002,
    "discovery_reserves": 7003,  # Time series reserve estimates
    "discovery_resource": 7006,
    "discovery_resource_chrono": 7007,
    "discovery_resource_litho": 7008,

    # Field details
    "field_activity_status": 7101,
    "field_area": 7102,
    "field_description": 7103,
    "field_investment_yearly": 7104,
    "field_licensee": 7105,
    "field_operator": 7106,
    "field_owner": 7107,
    "field_production_monthly": 7108,
    "field_production_yearly": 7109,
    "field_reserves": 7110,
    "field_in_place_volumes": 7111,
    "field_pipeline_transport": 7113,
    "field_facility_transport": 7114,

    # CO2 storage
    "csd_injection": 9001,
}


# =============================================================================
# FactMaps Layers (display-oriented, pre-filtered views)
# =============================================================================
FACTMAPS_LAYERS = {
    # Wellbores by category
    "wellbore_all": 201,
    "wellbore_exploration_active": 203,
    "wellbore_exploration": 204,
    "wellbore_development": 205,
    "wellbore_other": 206,
    "wellbore_co2": 207,

    # Facilities
    "facility_in_place": 304,
    "facility_not_in_place": 306,
    "facility_all": 307,
    "pipeline": 311,

    # Seismic surveys by status
    "seismic_pending": 403,
    "seismic_planned": 404,
    "seismic_ongoing": 405,
    "seismic_paused": 406,
    "seismic_cancelled": 407,
    "seismic_finished": 421,

    # EM surveys
    "em_pending": 409,
    "em_planned": 410,
    "em_ongoing": 411,
    "em_paused": 412,
    "em_cancelled": 413,
    "em_finished": 422,

    # Other surveys
    "survey_all": 420,
    "other_survey_pending": 415,
    "other_survey_planned": 416,
    "other_survey_ongoing": 417,
    "other_survey_paused": 418,
    "other_survey_cancelled": 419,
    "other_survey_finished": 423,

    # Discoveries and fields
    "field_by_status": 502,
    "discovery_active": 503,
    "discovery_all": 504,
    "discovery_history": 505,

    # Plays
    "play": 540,

    # Licensing
    "apa_gross": 603,
    "apa_open": 604,

    # Administrative
    "blocks": 802,
    "quadrants": 803,
    "sub_areas": 804,
}


# =============================================================================
# Dataset Categories (for organized syncing)
# =============================================================================
ENTITY_DATASETS = [
    "discovery", "field", "wellbore", "play",
    "facility", "pipeline", "licence", "block", "quadrant"
]

SUPPORTING_DATASETS = [
    # Stratigraphy
    "strat_chrono", "strat_litho", "strat_litho_wellbore", "strat_litho_wellbore_core",
    # Discovery
    "discovery_reserves", "discovery_resource", "discovery_operator", "discovery_area",
    # Field
    "field_reserves", "field_description", "field_activity_status", "field_area",
    "field_operator", "field_owner", "field_in_place_volumes",
    # Wellbore
    "wellbore_casing", "wellbore_core_photo", "wellbore_dst", "wellbore_history",
    "wellbore_chrono_strat", "wellbore_litho_strat", "wellbore_coordinates",
    "wellbore_document", "wellbore_log", "wellbore_mud",
    "wellbore_exploration_all", "wellbore_development_all",
    # Facility
    "facility_history",
    # Company & Licensing
    "company", "licence_licensee_history", "licence_operator_history", "licensing_activity",
]

PRODUCTION_DATASETS = [
    "field_production_monthly", "field_production_yearly",
    "field_investment_yearly", "field_pipeline_transport", "field_facility_transport",
]


# =============================================================================
# Data not available in API (must use FactPages downloads)
# =============================================================================
FACTPAGES_ONLY = """
The following data is NOT available via the REST API and must be
downloaded from FactPages (https://factpages.sodir.no/):

- Prospects (prospect entities, geometries)
- Prospect estimates (risk/volume data, low/basis/high scenarios)
- Prospect-well relationships
- Prospect-play linkages
- Discovery-prospect linkages

These datasets are critical for exploration analysis but are only
available through FactPages CSV/Excel downloads.
"""
