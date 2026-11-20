"""Grid TERAGConfig factory: the accepted run's fixed resource values.

The research construction (``cfg_from_meta``) validated the effective config
against the accepted run's recorded ``algorithm_parameters``; those constants
are mirrored in ``terag_release.paper_profile.PAPER_TERAG`` (the single
public source of truth, already validated by tests/test_config_binding.py
against the exported code). This factory applies them to a pair of schema
CSVs the same way the BIRD factory does. Supervision is injected by the
caller (``terag_release.supervision.inject_supervision``) before the engine
is constructed, exactly like the research Job.
"""
from terag.engines.bird_runtime.config import (
    TERAGConfig, DataConfig, IndexConfig, BM25Config, FieldWeightsConfig,
    GraphConfig, FeatureConfig, AnnotationsConfig, AblationConfig,
    OutputConfig, TokenizerConfig, PatternMiningConfig)


def make_grid_config(field_csv, table_csv, artifacts_dir,
                     language='chinese'):
    """Per-dataset TERAGConfig with the accepted grid resource values."""
    return TERAGConfig(
        seed=42,
        data=DataConfig(dataset='private', field_csv=str(field_csv),
                        table_csv=str(table_csv), qa_csv=''),
        index=IndexConfig(
            bm25=BM25Config(k1=1.5, b=0.75),
            field_weights=FieldWeightsConfig(table_name=2.0, table_desc=1.5,
                                             field_name=1.0, field_desc=0.8,
                                             pattern=0.5),
            include_field_desc=True),
        graph=GraphConfig(parse_mode='ast'),
        pattern_mining=PatternMiningConfig(mode='keywords'),
        feature=FeatureConfig(),
        annotations=AnnotationsConfig(use_history=True, use_transfer=True,
                                      use_annotation_index=True),
        ablation=AblationConfig(),
        output=OutputConfig(artifacts_dir=str(artifacts_dir), save_graph=False,
                            save_index=False, save_patterns=False,
                            save_role_stats=False),
        tokenizer=TokenizerConfig(language=language),
    )
