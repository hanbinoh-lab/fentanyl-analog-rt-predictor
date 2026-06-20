# Raw data

This directory contains the source-data transcription and the curated structure-retention time (RT) table used for modeling. Experimental conditions and detailed structure-curation procedures are described in the associated article and Supporting Information.

## Files

| File                   | Contents                                                              | Rows | Columns                                                                                                        |
| ---------------------- | --------------------------------------------------------------------- | ---: | -------------------------------------------------------------------------------------------------------------- |
| `source_kim2025.csv` | Transcription of the published compound, mass-transition, and RT data |   94 | `compound`, `precursor_mz`, `quant_ion_mz`, `qual_ion_1_mz`, `qual_ion_2_mz`, `retention_time_min` |
| `base.csv`           | Final modeling table with curated canonical SMILES                    |   94 | `compound`, `smiles`, `retention_time_min`                                                               |

The source study reports 93 fentanyl analogs. The modeling table contains 94 records because beta-hydroxy-3-methylfentanyl has two resolved RT assignments, labeled A (13.8 min) and B (14.5 min). Precursor- and product-ion mass-to-charge ratios were used for structure-consistency checks but were not used as model predictors. The CSV transcription contains the reported RT values but does not include the relative standard deviations or collision energies reported in the source publication.

## Source

Kim J, Seo S, Jeong C, Bae E, Lee D, Kim J, et al. Serially coupled columns enhance rapid separation and predictive interaction understanding of 93 fentanyl analogs. Anal Chim Acta. 2025;1336:343479. https://doi.org/10.1016/j.aca.2024.343479.

When using the reported RT or mass-transition data, cite the source article above. When using the curated structure-RT dataset (`base.csv`), also cite the archived release of this repository. The source article and its Supporting Information remain subject to the publisher's terms of use.
