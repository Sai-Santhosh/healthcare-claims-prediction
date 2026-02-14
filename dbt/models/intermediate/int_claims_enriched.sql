-- =============================================================================
-- Intermediate Model: int_claims_enriched
-- Description: Enrich claims with derived features and categorizations
-- =============================================================================

{{
    config(
        materialized='ephemeral',
        tags=['intermediate']
    )
}}

with staged_claims as (
    select * from {{ ref('stg_claims') }}
),

enriched as (
    select
        -- Original columns
        claim_id_key,
        member_state,
        member_county,
        age,
        gender_code,
        form_type,
        service_status,
        product_type,
        place_of_service,
        primary_diagnosis,
        secondary_diagnosis,
        procedure_code,
        amt_billed,
        amt_paid,
        amt_deductible,
        amt_coinsurance,
        length_of_stay,
        quantity,
        service_date,
        _loaded_at,
        
        -- Derived: Age bands
        case
            when age < 18 then '0-17'
            when age < 35 then '18-34'
            when age < 50 then '35-49'
            when age < 65 then '50-64'
            else '65+'
        end as age_band,
        
        -- Derived: Gender description
        case
            when gender_code = 'M' then 'Male'
            when gender_code = 'F' then 'Female'
            else 'Unknown'
        end as gender_description,
        
        -- Derived: ICD category (first character)
        upper(left(primary_diagnosis, 1)) as icd_category,
        
        -- Derived: Payment ratio
        case 
            when amt_billed > 0 then round(amt_paid / amt_billed, 4)
            else 0 
        end as payment_ratio,
        
        -- Derived: Out-of-pocket amount
        amt_deductible + amt_coinsurance as out_of_pocket,
        
        -- Derived: Net amount
        amt_paid - amt_deductible - amt_coinsurance as net_amount,
        
        -- Derived: High-cost flag
        case when amt_paid > 1000 then true else false end as is_high_cost,
        
        -- Derived: Number of diagnoses
        case 
            when secondary_diagnosis is not null then 2 
            else 1 
        end as num_diagnoses,
        
        -- Derived: State residency
        case 
            when member_state = 'NH' then 'In-State'
            else 'Out-of-State'
        end as residency_type
        
    from staged_claims
)

select * from enriched
