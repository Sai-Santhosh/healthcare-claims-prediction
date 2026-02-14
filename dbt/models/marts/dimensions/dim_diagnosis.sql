-- =============================================================================
-- Dimension: dim_diagnosis
-- Description: Diagnosis dimension with ICD codes and categories
-- =============================================================================

{{
    config(
        materialized='table',
        unique_key='diagnosis_key',
        tags=['dimension', 'diagnosis']
    )
}}

with claims_data as (
    select * from {{ ref('int_claims_enriched') }}
),

diagnosis_codes as (
    select distinct
        primary_diagnosis as icd_code,
        icd_category
    from claims_data
    where primary_diagnosis is not null
    
    union
    
    select distinct
        secondary_diagnosis as icd_code,
        upper(left(secondary_diagnosis, 1)) as icd_category
    from claims_data
    where secondary_diagnosis is not null
),

categorized as (
    select
        icd_code,
        icd_category,
        -- ICD-10 category descriptions
        case icd_category
            when 'A' then 'Infectious and parasitic diseases'
            when 'B' then 'Infectious and parasitic diseases'
            when 'C' then 'Neoplasms'
            when 'D' then 'Blood diseases and immune disorders'
            when 'E' then 'Endocrine and metabolic diseases'
            when 'F' then 'Mental and behavioral disorders'
            when 'G' then 'Nervous system diseases'
            when 'H' then 'Eye and ear diseases'
            when 'I' then 'Circulatory system diseases'
            when 'J' then 'Respiratory system diseases'
            when 'K' then 'Digestive system diseases'
            when 'L' then 'Skin diseases'
            when 'M' then 'Musculoskeletal diseases'
            when 'N' then 'Genitourinary system diseases'
            when 'O' then 'Pregnancy and childbirth'
            when 'P' then 'Perinatal conditions'
            when 'Q' then 'Congenital abnormalities'
            when 'R' then 'Symptoms and abnormal findings'
            when 'S' then 'Injury and external causes'
            when 'T' then 'Injury and external causes'
            when 'V' then 'External causes of morbidity'
            when 'W' then 'External causes of morbidity'
            when 'X' then 'External causes of morbidity'
            when 'Y' then 'External causes of morbidity'
            when 'Z' then 'Health status and contact factors'
            else 'Other/Unknown'
        end as category_description
    from diagnosis_codes
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['icd_code']) }} as diagnosis_key,
        icd_code,
        icd_category,
        category_description,
        current_timestamp() as _loaded_at
    from categorized
)

select * from final
