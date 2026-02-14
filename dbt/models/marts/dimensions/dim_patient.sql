-- =============================================================================
-- Dimension: dim_patient
-- Description: Patient dimension with demographic attributes
-- SCD Type: 2 (slowly changing dimension)
-- =============================================================================

{{
    config(
        materialized='table',
        unique_key='patient_key',
        tags=['dimension', 'patient']
    )
}}

with claims_data as (
    select * from {{ ref('int_claims_enriched') }}
),

patient_attributes as (
    select distinct
        member_state,
        member_county,
        age_band,
        gender_code,
        gender_description,
        residency_type
    from claims_data
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['member_state', 'member_county', 'age_band', 'gender_code']) }} as patient_key,
        member_state,
        member_county,
        age_band,
        gender_code,
        gender_description,
        residency_type,
        current_timestamp() as valid_from,
        cast(null as timestamp) as valid_to,
        true as is_current
    from patient_attributes
)

select * from final
