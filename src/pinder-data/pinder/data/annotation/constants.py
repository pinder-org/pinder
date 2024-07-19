from __future__ import annotations

RCSB_GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"

TYPE_PATTERNS: dict[str, list[str]] = {
    "ecod": ["ECOD"],
    "scop": [
        "SCOP",
        "SCOP2",
        "SCOP2B_SUPERFAMILY",
        "SCOP2_FAMILY",
        "SCOP2_SUPERFAMILY",
    ],
    "outlier": [
        "STEREO_OUTLIER",
        "BOND_OUTLIER",
        "RAMACHANDRAN_OUTLIER",
        "MOGUL_ANGLE_OUTLIER",
        "MOGUL_BOND_OUTLIER",
        "RSCC_OUTLIER",
        "RSRZ_OUTLIER",
        "ANGLE_OUTLIER",
        "ROTAMER_OUTLIER",
    ],
    "sabdab": [
        "SABDAB_ANTIBODY_HEAVY_CHAIN_SUBCLASS",
        "SABDAB_ANTIBODY_LIGHT_CHAIN_SUBCLASS",
        "SABDAB_ANTIBODY_LIGHT_CHAIN_TYPE",
    ],
    "occupancy": ["ZERO_OCCUPANCY_RESIDUE_XYZ", "ZERO_OCCUPANCY_ATOM_XYZ"],
    "unobserved": ["UNOBSERVED_ATOM_XYZ", "UNOBSERVED_RESIDUE_XYZ"],
    "cath": ["CATH"],
    "binding_site": ["BINDING_SITE"],
    "asa": ["ASA"],
    "other": [
        "UNASSIGNED_SEC_STRUCT",
        "HELIX_P",
        "SHEET",
        "MEMBRANE_SEGMENT",
        "CIS-PEPTIDE",
    ],
}


GRAPHQL_ANNOTATION_QUERY = """
query annotations($id: String!) {
  entry(entry_id: $id) {
    rcsb_id
    entry {
      id
    }
    rcsb_comp_model_provenance {
      source_url
      entry_id
      source_db
    }
    rcsb_associated_holdings {
      rcsb_repository_holdings_current_entry_container_identifiers {
        assembly_ids
      }
      rcsb_repository_holdings_current {
        repository_content_types
      }
    }
    rcsb_external_references {
      id
      type
      link
    }
    rcsb_entry_container_identifiers {
      emdb_ids
    }
    pdbx_database_status {
      pdb_format_compatible
    }
    struct {
      title
    }
    polymer_entities {
      entity_poly {
        rcsb_entity_polymer_type
        pdbx_seq_one_letter_code_can
      }
      rcsb_polymer_entity_container_identifiers {
        entry_id
        entity_id
        asym_ids
        auth_asym_ids
      }
      rcsb_polymer_entity_feature {
        type
        feature_id
        name
        provenance_source
        assignment_version
        feature_positions {
          end_seq_id
          beg_seq_id
        }
        additional_properties {
          name
          values
        }
        description
      }
      rcsb_id
      chem_comp_nstd_monomers {
        rcsb_id
        chem_comp {
          mon_nstd_parent_comp_id
        }
        rcsb_chem_comp_annotation {
          type
          annotation_id
          name
        }
        rcsb_chem_comp_related {
          resource_accession_code
          resource_name
        }
      }
      pfams {
        rcsb_id
        rcsb_pfam_accession
        rcsb_pfam_identifier
        rcsb_pfam_comment
        rcsb_pfam_description
        rcsb_pfam_seed_source
      }
      uniprots {
        rcsb_id
        rcsb_uniprot_annotation {
          annotation_id
          type
          name
          description
          assignment_version
          annotation_lineage {
            id
            name
            depth
          }
          additional_properties {
            name
            values
          }
        }
      }
      rcsb_polymer_entity_annotation {
        annotation_id
        type
        name
        description
        assignment_version
        additional_properties {
          name
          values
        }
        annotation_lineage {
          id
          name
          depth
        }
      }
      rcsb_polymer_entity {
        pdbx_description
        rcsb_enzyme_class_combined {
          ec
          provenance_source
        }
      }
      rcsb_entity_source_organism {
        ncbi_taxonomy_id
        common_name
        scientific_name
        ncbi_scientific_name
      }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers {
          asym_id
          entity_id
          auth_asym_id
          auth_to_entity_poly_seq_mapping
        }
        rcsb_polymer_instance_feature {
          ordinal
          type
          feature_id
          name
          provenance_source
          assignment_version
          feature_positions {
            end_seq_id
            beg_seq_id
          }
          reference_scheme
          additional_properties {
            name
            values
          }
        }
        rcsb_polymer_instance_annotation {
          annotation_id
          type
          assignment_version
          name
          description
          provenance_source
          annotation_lineage {
            id
            name
            depth
          }
        }
        rcsb_polymer_instance_feature_summary {
          coverage
          type
          count
        }
      }
    }
    exptl {
      method
    }
    nonpolymer_entities {
      nonpolymer_comp {
        chem_comp {
          id
        }
      }
      nonpolymer_entity_instances {
        rcsb_nonpolymer_instance_validation_score {
          ranking_model_fit
          ranking_model_geometry
          is_subject_of_investigation
        }
      }
    }
    assemblies {
      rcsb_assembly_feature {
        assignment_version
        description
        feature_id
        name
        provenance_source
        type
        feature_positions {
          asym_id
          beg_seq_id
          struct_oper_list
        }
        additional_properties {
          name
          values
        }
      }
    }
  }
}
"""
