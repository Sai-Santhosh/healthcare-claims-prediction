-- =============================================================================
-- Staging Model: stg_claims
-- Description: Clean and standardize raw claims data from source
-- =============================================================================

{{
    config(
        materialized='view',
        tags=['staging', 'claims']
    )
}}

with source as (
    select * from {{ source('raw', 'claims_raw') }}
),

cleaned as (
    select
        -- Primary key
        claim_id_key,
        
        -- Patient demographics
        member_state,
        member_county,
        age,
        sex as gender_code,
        
        -- Claim attributes
        form_type,
        sv_stat as service_status,
        product_type,
        pos as place_of_service,
        
        -- Diagnosis codes
        icd_diag_01 as primary_diagnosis,
        icd_diag_02 as secondary_diagnosis,
        
        -- Procedure codes
        cpt as procedure_code,
        
        -- Financial amounts
        coalesce(amt_billed, 0) as amt_billed,
        coalesce(amt_paid, 0) as amt_paid,
        coalesce(amt_deduct, 0) as amt_deductible,
        coalesce(amt_coins, 0) as amt_coinsurance,
        
        -- Utilization
        coalesce(client_los, 0) as length_of_stay,
        coalesce(qty, 1) as quantity,
        
        -- Dates
        service_date,
        
        -- Metadata
        current_timestamp() as _loaded_at
        
    from source
    where 
        -- Filter out invalid records
        claim_id_key is not null
        and amt_billed >= 0
        and amt_paid >= 0
)

select * from cleaned
