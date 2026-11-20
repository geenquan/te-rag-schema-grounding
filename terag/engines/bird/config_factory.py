"""Frozen per-database TERAGConfig factory for the BIRD engines.

Consolidates the make_bird_config construction of the final research runs
(repair_rerun.py::_make_dev_config and search/h/h_engine_driver.py::
make_bird_config, identical fixed values) into one public factory. Every
value is frozen paper identity; nothing here is tuned.

The schema_dir parameter points at a per-database directory containing
field_schema.csv and table_schema.csv (the released frozen copies live
under data/bird/processed/schema/<db_id>/).
"""

from terag.engines.bird_runtime.config import (TERAGConfig, DataConfig, IndexConfig,
                                               BM25Config, FieldWeightsConfig,
                                               GraphConfig, FeatureConfig,
                                               AnnotationsConfig, AblationConfig,
                                               OutputConfig, TokenizerConfig,
                                               PatternMiningConfig,
                                               BirdDatasetConfig)


def make_bird_config(db_id, schema_dir, artifacts_dir,
                     language='english', dialect='sqlite'):
    """Build the frozen per-database TERAGConfig.

    Args:
        db_id: BIRD database id (documentation/provenance only; the CSVs
            under schema_dir define the database).
        schema_dir: directory holding the db's field_schema.csv and
            table_schema.csv.
        artifacts_dir: output directory for engine artifacts (cache).
        language: tokenizer language (frozen 'english' for BIRD).
        dialect: SQL dialect (frozen 'sqlite' for BIRD).

    Returns:
        TERAGConfig with the frozen final-paper values: seed 42, BM25
        k1=1.5 b=0.75, field weights 2.0/1.5/1.0/0.8/0.5,
        include_field_desc=True, PatternMiningConfig(mode='generic'),
        annotations all on, and BirdDatasetConfig(sql_dialect=dialect,
        use_evidence=False, use_database_description=True,
        use_value_description=True).
    """
    from pathlib import Path
    schema_dir = Path(schema_dir)
    cfg = TERAGConfig(
        seed=42,
        data=DataConfig(dataset='bird', field_csv=str(schema_dir / 'field_schema.csv'),
                        table_csv=str(schema_dir / 'table_schema.csv'), qa_csv=''),
        index=IndexConfig(
            bm25=BM25Config(k1=1.5, b=0.75),
            field_weights=FieldWeightsConfig(table_name=2.0, table_desc=1.5,
                                             field_name=1.0, field_desc=0.8,
                                             pattern=0.5),
            include_field_desc=True),
        graph=GraphConfig(),
        pattern_mining=PatternMiningConfig(mode='generic'),
        feature=FeatureConfig(),
        annotations=AnnotationsConfig(use_history=True, use_transfer=True,
                                      use_annotation_index=True),
        ablation=AblationConfig(),
        output=OutputConfig(artifacts_dir=str(artifacts_dir), save_graph=False,
                            save_index=False, save_patterns=False,
                            save_role_stats=False),
        tokenizer=TokenizerConfig(language=language),
    )
    cfg.data.bird = BirdDatasetConfig(sql_dialect=dialect, use_evidence=False,
                                      use_database_description=True,
                                      use_value_description=True)
    return cfg
