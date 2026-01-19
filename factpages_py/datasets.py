"""
Dataset definitions for the Sodir REST API.

This module contains the mapping of dataset names to API layer/table IDs.
The API is built on ArcGIS FeatureServer and provides two main services:

1. DataService - Clean tabular data with consistent naming
2. FactMaps - Display-oriented layers with pre-filtered views

Note: Production data (monthly/yearly) is NOT available via this API.
For production data, use the Sodir FactPages downloads.
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
    "structural_elements": 2000,  # str_structuralelements
    "domes": 2001,  # str_domes
    "faults_boundaries": 2002,  # str_faultsandboundaries
    "sediment_boundaries": 2004,  # str_sedimentboundaries

    # Licensing areas
    "licence": 3000,
    "licence_area_poly_hst": 3002,
    "licence_document_area": 3006,  # licence_document_area_poly
    "licence_area_count": 3011,  # licence_area_licenced_count
    "apa_gross": 3102,  # licensing_activity_apa_gross
    "apa_open": 3103,  # licensing_activity_apa_open
    "announced_blocks_history": 3104,  # licensing_activity_announced_blocks_hst
    "announced_history": 3106,  # licencing_activity_announced_hst
    "apa_gross_history": 3107,  # licensing_activity_apa_gross_hst
    "afex_area": 3200,
    "afex_area_history": 3201,  # afex_area_area_poly_hst
    "business_arrangement_area": 3300,  # bsns_arr_area
    "business_arrangement_history": 3301,  # bsns_arr_area_area_poly_hst

    # Seismic surveys
    "seismic_acquisition": 4000,  # seis_acquisition
    "seismic_acquisition_poly": 4008,  # seis_acquisition_poly_total
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
    "discovery_poly_hst": 7005,
    "field": 7100,
    "play": 7800,  # play_current_public

    # Seabed minerals
    "sbm_occurrence": 8001,  # sbm_occurence (typo in API)
    "sbm_play_resource_estimate": 8002,  # sbm_play_resource_estimate_area
}


# =============================================================================
# DataService Tables (no geometry)
# =============================================================================
TABLES = {
    # Company information
    "company": 1200,

    # Stratigraphy
    "strat_litho": 2100,
    "strat_litho_wellbore": 2101,
    "strat_litho_wellbore_core": 2102,
    "strat_chrono": 2200,

    # Licence details
    "licence_additional_area": 3001,
    "licence_transfer_hst": 3003,
    "licence_document": 3005,
    "licence_licensee_hst": 3007,
    "licence_operator_hst": 3008,
    "licence_phase_hst": 3009,
    "licence_task": 3010,
    "licensing_activity": 3100,

    # Business arrangement details
    "business_arrangement_operator": 3302,  # bsns_arr_area_operator
    "business_arrangement_licensee_hst": 3304,  # bsns_arr_area_licensee_hst
    "business_arrangement_transfer_hst": 3305,  # bsns_arr_area_transfer_hst

    # Petroleum register
    "petreg_licence": 3400,
    "petreg_licence_licensee": 3401,
    "petreg_licence_message": 3402,
    "petreg_licence_operator": 3403,  # petreg_licence_oper

    # Seismic details
    "seismic_acquisition_company": 4001,  # seis_acquisition_company_doing
    "seismic_acquisition_format": 4002,  # seis_acquisition_data_format
    "seismic_acquisition_fishery": 4003,
    "seismic_acquisition_for_company": 4004,
    "seismic_acquisition_licence": 4005,
    "seismic_acquisition_licences": 4006,
    "seismic_acquisition_polygon": 4009,
    "seismic_acquisition_progress": 4011,  # seis_acquisition_progress_notes
    "seismic_acquisition_scientific": 4012,  # seis_acquisition_scientific_survey
    "seismic_acquisition_vessel": 4013,
    "seismic_acquisition_weekly_done": 4014,
    "seismic_acquisition_weekly_plan": 4015,

    # Wellbore details
    "wellbore_casing": 5001,  # wellbore_casing_and_lot
    "wellbore_co2": 5002,
    "wellbore_core": 5003,
    "wellbore_core_photo": 5004,
    "wellbore_core_photo_aggr": 5005,
    "wellbore_cutting": 5006,
    "wellbore_document": 5007,
    "wellbore_dst": 5008,
    "wellbore_formation_top": 5009,
    "wellbore_log": 5011,
    "wellbore_mud": 5012,
    "wellbore_oil_sample": 5013,
    "wellbore_paly_slide": 5014,
    "wellbore_thin_section": 5015,
    "wellbore_history": 5050,

    # Facility details
    "facility_function": 6001,
    "tuf": 6200,
    "tuf_operator_hst": 6201,
    "tuf_owner_hst": 6202,

    # Discovery details
    "discovery_description": 7001,
    "discovery_extends_into": 7002,
    "discovery_licensee_hst": 7003,
    "discovery_operator_hst": 7006,
    "discovery_owner_hst": 7007,
    "discovery_reserves": 7008,

    # Field details
    "field_activity_status_hst": 7101,
    "field_description": 7102,
    "field_discoveries_incl_hst": 7103,
    "field_extends_into": 7104,
    "field_image": 7106,
    "field_investment_expected": 7107,
    "field_licensee_hst": 7108,
    "field_operator_hst": 7110,
    "field_owner_hst": 7111,
    "field_pdo_hst": 7112,
    "field_reserves": 7113,
    "field_reserves_company": 7114,

    # Profiles
    "profiles": 7300,

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
    "discovery_reserves", "discovery_description", "discovery_operator_hst",
    "discovery_licensee_hst", "discovery_owner_hst",
    # Field
    "field_reserves", "field_description", "field_activity_status_hst",
    "field_operator_hst", "field_owner_hst", "field_licensee_hst",
    "field_reserves_company", "field_investment_expected",
    # Wellbore
    "wellbore_casing", "wellbore_core_photo", "wellbore_dst", "wellbore_history",
    "wellbore_formation_top", "wellbore_document", "wellbore_log", "wellbore_mud",
    "wellbore_core", "wellbore_cutting", "wellbore_oil_sample",
    # Facility
    "facility_function",
    # Company & Licensing
    "company", "licence_licensee_hst", "licence_operator_hst", "licensing_activity",
]

# Note: Production data is NOT available via this API
# Use Sodir FactPages downloads for production data
PRODUCTION_DATASETS = []


# =============================================================================
# Metadata Service (table and column descriptions)
# =============================================================================
METADATA_BASE_URL = "https://factmaps.sodir.no/api/rest/services/DataService/Metadata/FeatureServer"

METADATA_TABLES = {
    "table_descriptions": 10,      # Human-readable descriptions of each table
    "attribute_descriptions": 11,  # Human-readable descriptions of each column/field
}
