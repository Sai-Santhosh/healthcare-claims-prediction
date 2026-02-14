-- =============================================================================
-- Fact Table: fact_claims
-- Description: Claims fact table with measures and dimension keys
-- Grain: One row per claim line item
-- =============================================================================

{{
    config(
        materialized='incremental',
        unique_key='claim_fact_key',
        incremental_strategy='merge',
        tags=['fact', 'claims']
    )
}}

with claims_enriched as (
    select * from {{ ref('int_claims_enriched') }}
    {% if is_incremental() %}
    where _loaded_at > (select max(_loaded_at) from {{ this }})
    {% endif %}
),

dim_patient as (
    select * from {{ ref('dim_patient') }}
),

dim_diagnosis as (
    select * from {{ ref('dim_diagnosis') }}
),

-- Join with dimensions
with_dimensions as (
    select
        c.claim_id_key,
        
        -- Dimension keys
        p.patient_key,
        d.diagnosis_key as primary_diagnosis_key,
        
        -- Degenerate dimensions
        c.form_type,
        c.service_status,
        c.product_type,
        c.place_of_service,
        c.procedure_code,
        
        -- Measures
        c.amt_billed,
        c.amt_paid,
        c.amt_deductible,
        c.amt_coinsurance,
        c.out_of_pocket,
        c.net_amount,
        c.length_of_stay,
        c.quantity,
        
        -- Derived measures
        c.payment_ratio,
        c.is_high_cost,
        c.num_diagnoses,
        
        -- Date dimension (would link to dim_date)
        c.service_date,
        
        -- Metadata
        c._loaded_at
        
    from claims_enriched c
    left join dim_patient p
        on c.member_state = p.member_state
        and c.member_county = p.member_county
        and c.age_band = p.age_band
        and c.gender_code = p.gender_code
        and p.is_current = true
    left join dim_diagnosis d
        on c.primary_diagnosis = d.icd_code
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['claim_id_key', 'service_date']) }} as claim_fact_key,
        *
    from with_dimensions
)

select * from final
